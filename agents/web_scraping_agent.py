import os
from typing import Optional

from bs4 import BeautifulSoup
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

CONFIG = {
    "FIREFOX_BINARY": f"{parent_dir}/firefox/firefox-bin",
    "GECKODRIVER_PATH": f"{parent_dir}/geckodriver",
}
class WebScraper:
    def __init__(self):
        self.options = self._configure_selenium()
        self.service = Service(CONFIG['GECKODRIVER_PATH'])

    def _configure_selenium(self) -> Options:
        """Configure Selenium Firefox options"""
        options = Options()
        options.add_argument("--headless")
        options.binary_location = CONFIG["FIREFOX_BINARY"]
        return options

    def _scrape_website(self, url: str) -> Optional[str]:
        """Scrape website content using Selenium with explicit waits"""
        print(f"Starting scrape for URL: {url}", )
        try:
            driver = Firefox(
                service=self.service,
                options=self.options
            )
            driver.get(url)

            # Wait for page to load using explicit wait
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'body'))
                )
                print(f"Page: {url} scraped successfully")
            except TimeoutException:
                print(f"Page: {url} - load timeout, proceeding with current content")
            # time.sleep(10)

            html = driver.page_source
            driver.quit()
            print(f"Page content retrieved {len(html)} characters)")
            return html

        except Exception as e:
            print(f"Scraping failed for {url}: {str(e)}")
            return None

    def _process_html_content(self, html: str) -> str:
        """Process HTML content with BeautifulSoup"""
        if not html:
            print("Received empty HTML content")
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")

            unwanted_tags = soup(["script", "style"])
            for tag in unwanted_tags:
                tag.decompose()

            text = soup.get_text(separator="\n")
            cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]

            print(f"Lines after cleaning - {len(cleaned_lines)}")

            return "\n".join(cleaned_lines)
        except Exception as e:
            print(f"HTML processing failed: {str(e)}")
            return ""

    def run(self, url: str) -> str:
        """
        Scrapes a website and returns cleaned HTML content.

        Args:
            url: The URL to scrape.

        Returns:
            The cleaned HTML content as a string.
        """
        html = self._scrape_website(url)
        if html:
            return self._process_html_content(html)
        return ""
