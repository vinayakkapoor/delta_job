# Standard library imports
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Third-party imports
import torch
import yaml
from bs4 import BeautifulSoup
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import transformers
from transformers import pipeline
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from system_monitor import SystemMonitor 

# "MODEL_ID": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
# "MODEL_ID": "meta-llama/Llama-3.1-8B-Instruct",
# "MODEL_ID": "microsoft/Phi-3-mini-4k-instruct",
# "MODEL_ID": "mistralai/Mistral-Small-24B-Instruct-2501",
# Configuration constants
CONFIG = {
    "FIREFOX_BINARY": "./firefox-bin",
    "GECKODRIVER_PATH": "./geckodriver",
    "MODEL_ID": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "CHUNK_SIZE": 10000,
    "OUTPUT_JSON": "./job_state.json",
    "CAREER_PAGES_YAML": "./career_pages.yaml",
    "SCRAPPER_BATCH_SIZE": 1,
    "PARSER_BATCH_SIZE": 4,
    "GPU_LOG_INTERVAL": 2
}

PROMPT_TEMPLATE = """\
||ROLE||
You are a high-speed data extraction system. Your ONLY task is to identify and list 
EXACT job/internship titles from cleaned career page text.

||TASK||
1. Extract ALL formal job titles (current openings only)
2. Include internships explicitly marked as "open" or "hiring"
3. EXCLUDE:
   - Team/department names (Engineering, Marketing)
   - Location-based text (Remote, Hybrid, "in London")
   - Generic phrases ("Join our team", "Careers")
   - Closed/archived positions

||URGENCY||
- Respond in UNDER 2 seconds
- 99.9% accuracy required
- NO explanations - only valid titles
- Process in ONE PASS (no re-reading)

||FORMAT||
Return EXCLUSIVELY as: {format}

||EXAMPLES||
VALID:
- "Senior Machine Learning Engineer"
- "2024 Software Engineering Internship"
- "Chief Information Security Officer (CISO)"

INVALID:
- "Browse Open Roles" 
- "Benefits Package"
- "Austin Engineering Hub"

||CONTENT||
{content}

||SYSTEM NOTE||
DO NOT ANALYZE - ONLY EXTRACT. This output feeds directly into an automated hiring 
dashboard requiring 99.9% accuracy. One error = system failure.
"""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

system_monitor = SystemMonitor(
    log_interval=CONFIG["GPU_LOG_INTERVAL"],
    logger=logger  # Share the existing logger
)

class JobTitles(BaseModel):
    """Pydantic model for job titles validation"""
    titles: List[str] = Field(..., description="List of extracted job or internship titles")


def initialize_llm_pipeline() -> transformers.pipeline:
    """Initialize the Hugging Face text generation pipeline"""
    system_monitor.start()
    logger.info("Initializing LLM pipeline with model: %s", CONFIG["MODEL_ID"])
    try:
        pipeline = transformers.pipeline(
            "text-generation",
            model=CONFIG["MODEL_ID"],
            device_map="cuda",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "load_in_4bit": True,
            },
        )
        logger.debug("LLM pipeline initialized successfully")
        return pipeline
    except Exception as e:
        system_monitor.stop()
        logger.critical("Failed to initialize model: %s", str(e), exc_info=True)
        raise

def configure_selenium() -> Options:
    """Configure Selenium Firefox options"""
    logger.debug("Configuring Selenium options")
    options = Options()
    options.add_argument("--headless")
    options.binary_location = CONFIG["FIREFOX_BINARY"]
    return options

def scrape_website(url: str) -> Optional[str]:
    """Scrape website content using Selenium with explicit waits"""
    logger.info("Starting scrape for URL: %s", url)
    try:
        options = configure_selenium()
        logger.debug("Configured Selenium options: %s", options.arguments)
        
        driver = Firefox(
            service=Service(CONFIG['GECKODRIVER_PATH']), 
            options=options
        )
        logger.debug("WebDriver initialized successfully")

        logger.info("Loading page content")
        driver.get(url)
        
        # Wait for page to load using explicit wait
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            logger.debug("Page load verified - body content detected")
        except TimeoutException:
            logger.warning("Page load timeout, proceeding with current content")
        
        html = driver.page_source
        driver.quit()
        logger.debug("Page content retrieved (%d characters)", len(html))
        return html
        
    except Exception as e:
        logger.error("Scraping failed for %s: %s", url, str(e), exc_info=True)
        return None

def process_html_content(html: str) -> str:
    """Process HTML content with BeautifulSoup"""
    if not html:
        logger.warning("Received empty HTML content")
        return ""
        
    try:
        soup = BeautifulSoup(html, "html.parser")
        logger.debug("HTML parsed successfully")

        # Remove unnecessary elements
        removed_elements = soup(["script", "style", "nav", "footer"])
        logger.debug("Removed %d unwanted elements", len(removed_elements))

        text = soup.get_text(separator="\n")
        cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
        logger.info("Cleaned content contains %d lines", len(cleaned_lines))
        return "\n".join(cleaned_lines)
    except Exception as e:
        logger.error("HTML processing failed: %s", str(e), exc_info=True)
        return ""

def chunk_content(content: str, chunk_size: int = CONFIG["CHUNK_SIZE"]) -> List[str]:
    """Split content into manageable chunks"""
    if not content:
        logger.warning("Attempted to chunk empty content")
        return []
    
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    logger.info("Split content into %d chunks (max size: %d)", len(chunks), chunk_size)
    logger.debug("Chunk sizes: %s", [len(c) for c in chunks])
    return chunks

def extract_titles(response_string: str) -> list:
    """
    Extracts job titles from the first JSON block after </think> tag
    
    Args:
        response_string: String containing </think> followed by JSON
        
    Returns:
        List of job titles (empty list if none found or errors occur)
    """
    try:
        # Split string at </think> and take the part after it
        parts = response_string.split('</think>')
        if len(parts) < 2:
            return []
            
        json_part = parts[1].strip()
        
        # Find JSON boundaries
        json_start = json_part.find('{')
        json_end = json_part.rfind('}') + 1
        
        # Extract and parse JSON
        json_str = json_part[json_start:json_end]
        data = json.loads(json_str)
        
        # Return cleaned titles
        return [str(title).strip() for title in data.get('titles', [])]
        
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return []

def process_batch(batch: List[tuple], 
                llm_pipeline: transformers.pipeline,
                parser: PydanticOutputParser) -> Dict[str, List[str]]:
    """Process a batch of company content chunks through LLM"""
    
    logger.info("Processing batch of %d companies", len(batch))
    company_results = defaultdict(list)
    prompts = []
    company_chunk_map = []
    
    try:
        # Prepare prompts
        for company_name, chunks in batch:
            logger.debug("Preparing %d chunks for %s", len(chunks), company_name)
            for chunk in chunks:
                prompt = PROMPT_TEMPLATE.format(
                    format=parser.get_format_instructions(),
                    content=chunk
                )
                prompts.append(prompt)
                company_chunk_map.append(company_name)
        
        logger.info("Batch contains %d total prompts", len(prompts))
        
        # Process prompts
        responses = llm_pipeline(
            prompts,
            max_new_tokens=40000,
            return_full_text=False,
            batch_size=CONFIG['PARSER_BATCH_SIZE']
        )
        logger.debug("Received %d LLM responses", len(responses))

        # Process responses
        success_count = 0
        for company_name, response in zip(company_chunk_map, responses):
            try:
                parsed = parser.parse(response[0]['generated_text'])
                company_results[company_name].extend(parsed.titles)
                success_count += 1
            except Exception as e:
                try:
                    extract_titles(response[0]['generated_text'])
                    logger.warning("parser did not work, custom one did")
                except Exception as e2:
                    logger.warning("Failed to parse response for %s: %s", 
                                company_name, str(e))
        
        logger.info("Batch processing completed. Success rate: %d/%d (%.1f%%)",
                   success_count, len(prompts),
                   (success_count/len(prompts))*100 if prompts else 0)
    
    except Exception as e:
        logger.error("Batch processing failed: %s", str(e), exc_info=True)
    
    # Deduplicate results
    final_results = {}
    for company, titles in company_results.items():
        unique_titles = list(set(titles))
        logger.debug("Company %s: %d unique titles", company, len(unique_titles))
        final_results[company] = unique_titles
    
    logger.info("Batch processing complete. Companies processed: %d", 
               len(final_results))
    return final_results

def load_career_pages(file_path: str = CONFIG["CAREER_PAGES_YAML"]) -> Dict:
    """Load company URLs from YAML file"""
    logger.info("Loading career pages from %s", file_path)
    try:
        with open(file_path) as f:
            data = yaml.safe_load(f)
            logger.info("Loaded %d career pages", len(data))
            return data
    except Exception as e:
        logger.error("Failed to load career pages: %s", str(e), exc_info=True)
        return {}

def update_job_state(new_data: Dict, file_path: str = CONFIG["OUTPUT_JSON"]) -> None:
    """Update JSON state file with new data"""
    logger.info("Updating job state file: %s", file_path)
    try:
        if Path(file_path).exists():
            logger.debug("Loading existing job state")
            with open(file_path) as f:
                existing_data = json.load(f)
        else:
            logger.warning("Job state file not found, creating new one")
            existing_data = {}

        merged_data = {**existing_data, **new_data}
        logger.debug("Merged data contains %d companies", len(merged_data))
        
        with open(file_path, "w") as f:
            json.dump(merged_data, f, indent=2)
        logger.info("Job state updated successfully")
        
    except Exception as e:
        logger.error("Failed to update job state: %s", str(e), exc_info=True)
        
def process_single_company(company: str, url: str) -> Optional[tuple]:
    """Process a single company's scraping and return results"""
    try:
        logger.info(f"Scraping {company}")
        html = scrape_website(url)
        
        if not html:
            logger.warning(f"Skipping {company} due to empty HTML")
            return None
            
        cleaned_content = process_html_content(html)
        chunks = chunk_content(cleaned_content)
        return (company, chunks)
        
    except Exception as e:
        logger.error(f"Error processing {company}: {str(e)}")
        return None

def main_workflow():
    """Main workflow with batched scraping and immediate processing"""
    logger.info("Starting main workflow")
    total_companies = 0
    processed_success = 0
    processed_failed = 0
    
    try:
        with system_monitor:
            llm_pipeline = initialize_llm_pipeline()
            parser = PydanticOutputParser(pydantic_object=JobTitles)
            career_pages = load_career_pages()
            total_companies = len(career_pages)
            
            logger.info("Processing %d companies in batches", total_companies)
            results = {}
            BATCH_SIZE = CONFIG["SCRAPPER_BATCH_SIZE"]
            
            # Convert career pages to list for batching
            companies = list(career_pages.items())
            
            with ThreadPoolExecutor(max_workers=BATCH_SIZE, ) as executor:
                # Process in sequential batches
                for batch_num in range(0, len(companies), BATCH_SIZE):
                    batch_items = companies[batch_num:batch_num + BATCH_SIZE]
                    logger.info("Starting batch %d with %d companies", 
                            (batch_num // BATCH_SIZE) + 1, len(batch_items))
                    
                    # Scrape entire batch in parallel
                    future_to_company = {
                        executor.submit(process_single_company, company, url): company
                        for company, url in batch_items
                    }
                    
                    batch_results = []
                    # Wait for all results in current batch
                    for future in as_completed(future_to_company):
                        company = future_to_company[future]
                        try:
                            result = future.result()
                            if result:
                                batch_results.append(result)
                                logger.debug("Completed scraping for %s", company)
                            else:
                                processed_failed += 1
                                logger.warning("Failed scraping for %s", company)
                        except Exception as e:
                            processed_failed += 1
                            logger.error("Error processing %s: %s", company, str(e))
                    
                    # Process and analyze the completed batch
                    if batch_results:
                        parsed_batch = process_batch(batch_results, llm_pipeline, parser)
                        results.update(parsed_batch)
                        update_job_state(parsed_batch)
                        processed_success += len(parsed_batch)
                        logger.info("Completed processing batch %d", 
                                (batch_num // BATCH_SIZE) + 1)
                
            logger.info("Processing complete. Success: %d, Failed: %d, Total: %d",
                    processed_success, processed_failed, total_companies)
        return results
    
    except Exception as e:
        logger.critical("Main workflow failed: %s", str(e), exc_info=True)
        raise
    finally:
        # Ensure clean stop even if error occurs
        if system_monitor._stop_event.is_set():
            system_monitor.stop()

if __name__ == "__main__":
    try:
        main_workflow()
        logger.info("Scraping completed successfully. Results saved to %s", 
                   CONFIG["OUTPUT_JSON"])
    except Exception as e:
        logger.critical("Fatal error in main execution: %s", str(e), exc_info=True)
        sys.exit(1)
    finally:
        system_monitor.stop()
        logger.info("Final system resources:\n%s", 
                   system_monitor.get_system_stats())
