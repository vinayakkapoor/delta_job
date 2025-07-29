import ast

from agents.gemma_llm_agent import LLMAgent
from agents.web_scraping_agent import WebScraper

url = "https://www.path-robotics.com/who-we-are/careers/"
web_scraper = WebScraper()
llm_agent = LLMAgent()

job_titles_str = llm_agent.run(web_scraper.run(url))
try:
    job_titles = ast.literal_eval(job_titles_str)
    if not isinstance(job_titles, list):
        job_titles = [str(job_titles)]
except (ValueError, SyntaxError):
    job_titles = [job_titles_str.strip()]

print(job_titles)
