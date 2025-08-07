import ast
import yaml
import time
import json
import datetime
from threading import Thread, Lock
from typing import Callable

from agents.gemma_llm_agent import LLMAgent
from agents.web_scraping_agent import WebScraper

NUM_THREADS_SCRAPE = 5

web_scraper = WebScraper()
llm_agent = LLMAgent()
lock = Lock()


scraped_pages = []
job_titles = []
all_pages_scraped = False

def load_career_pages(file='career_pages_short.yaml') -> dict:
    with open(file, 'r') as yaml_file:
        career_pages = yaml.safe_load(yaml_file)
    return career_pages

def url_generator(career_page_urls: list[str]):
    for url in career_page_urls:
        yield url

def update_scraped_pages(url, web_scraper):
    print(f"scraping: {url}")
    page = web_scraper.run(url)
    with lock:
        scraped_pages.append(page)

def page_scrape_worker(url_gen: Callable) -> None:
    web_scraper = WebScraper()
    try:
        while True:
            url = next(url_gen)
            update_scraped_pages(url, web_scraper)
    except StopIteration:
        pass

def get_job_titles():
    with lock:
        if len(scraped_pages) != 0:
            page = scraped_pages.pop(0)
        else:
            page = ""
    if not page == "":        
        try:
            print("*** Running inference ***")
            t1 = time.time()
            job_titles_str = llm_agent.run(page)
            t2 = time.time()
            if job_titles_str == None:
                print("Rate limit hit, waiting for 1 minute before querying")
                time.sleep(65)
                t1 = time.time()
                job_titles_str = llm_agent.run(page)
                t2 = time.time()
            print(f"This instance of inference took: {t2-t1} sec")
            titles = ast.literal_eval(str(job_titles_str))
            return titles
        except Exception as e:
            print(f"Invalid output by llm {job_titles_str} \n\n {str(e)}")
    
    return []

def llm_inference_worker():
    while not all_pages_scraped or len(scraped_pages) > 0:
        titles = get_job_titles()
        with lock:
            job_titles.append(titles)
        time.sleep(2) # avoid rate limits


def main():
    career_pages = load_career_pages()
    
    companies = career_pages.keys()
    career_page_urls = career_pages.values()

    url_gen = url_generator(career_page_urls)
    
    scrape_threads = []
    for i in range(NUM_THREADS_SCRAPE):
        t = Thread(target=page_scrape_worker, args=(url_gen, ))
        scrape_threads.append(t)
    for t in scrape_threads:
        t.start()
    
    llm_thread = Thread(target=llm_inference_worker)
    llm_thread.start()
    
    for t in scrape_threads:
        t.join()

    global all_pages_scraped
    all_pages_scraped = True

    llm_thread.join()

    company_jobs = dict(zip(companies, job_titles))
    file_path = f"job_titles_{datetime.datetime.now().isoformat()}.json"
    with open(file_path, "w") as json_file:
        json.dump(company_jobs, json_file, indent=4)
    print(company_jobs)


if __name__ == "__main__":
    main()

