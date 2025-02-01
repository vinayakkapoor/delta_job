# delta_job
Scrape career pages to extract open job / internship positions using LLMs. Compare them to previous openings to know if a company has posted new jobs.

## Usage

Create a virual environment using conda / virtualenv and install the necessary packages using `pip install -r requirements.txt`

Download Firefox using `wget https://ftp.mozilla.org/pub/firefox/releases/133.0/linux-x86_64/en-US/firefox-133.0.tar.bz2`

And extract `tar xjf firefox-*.tar.bz2`

Scrape, clean, parse using
`python run.py`

This will save a **job_lists.json**.

Compare two json files using the compare scripts provided

## Notes

- Try with models of your choice! DeepSeek-R1-Distill-Llama-8B gave me the best results :D. Requires about 6-8GB VRAM with 4 bit quantisation (enabled in the script)
- The Firefox binary and driver provided are for linuxx86_64. Replace them as per your machine requirements. Keep in mind that the version of Firefox and geckodriver should be same for smooth scraping
- delta_job = delta (change in) job
