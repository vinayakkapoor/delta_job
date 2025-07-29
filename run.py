
import asyncio
import yaml
import json
import ast
from datetime import datetime

from agents.gemma_llm_agent import LLMAgent
from agents.web_scraping_agent import WebScraper

# Dictionary to store job titles for each company
company_jobs = {}

# Initialize agents
web_scraper = WebScraper()
llm_agent = LLMAgent()


async def scrape_company(company_name, url):
    """Asynchronously scrapes a company's career page, processes it with an LLM,
    and stores the results."""
    print(f"Starting to scrape: {company_name}")
    try:
        scraped_content = web_scraper.run(url)
        if scraped_content:
            job_titles_str = llm_agent.run(scraped_content)
            try:
                job_titles = ast.literal_eval(job_titles_str)
                if not isinstance(job_titles, list):
                    job_titles = [str(job_titles)]
            except (ValueError, SyntaxError):
                job_titles = [job_titles_str.strip()]
            company_jobs[company_name] = job_titles
            print(f"Successfully scraped and processed: {company_name}")
        else:
            print(f"No content scraped from: {company_name}")
            company_jobs[company_name] = []
    except Exception as e:
        print(f"An error occurred while processing {company_name}: {e}")
        company_jobs[company_name] = ["Error during scraping"]


async def save_results_periodically():
    """Periodically saves the company_jobs dictionary to a file."""
    while True:
        await asyncio.sleep(60)  # Wait for 60 seconds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"job_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(company_jobs, f, indent=4)
        print(f"Results saved to {filename}")


async def main():
    """Main function to orchestrate the scraping and saving tasks."""
    # Load career pages from YAML file
    with open("career_pages.yaml", "r") as f:
        career_pages = yaml.safe_load(f)

    # Start the periodic saving task
    save_task = asyncio.create_task(save_results_periodically())

    # Create and queue scraping tasks with a delay
    scraping_tasks = []
    for company, url in career_pages.items():
        task = asyncio.create_task(scrape_company(company, url))
        scraping_tasks.append(task)
        await asyncio.sleep(3)  # 3-second delay between starting each company scrape

    # Wait for all scraping tasks to complete
    await asyncio.gather(*scraping_tasks)

    # At the end, do a final save
    final_filename = "final_job_results.json"
    with open(final_filename, "w") as f:
        json.dump(company_jobs, f, indent=4)
    print(f"Final results saved to {final_filename}")

    # Stop the periodic saving task
    save_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
