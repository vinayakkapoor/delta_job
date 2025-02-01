# delta_job
Scrape career pages to extract open job / internship positions. Compare them to previous openings to know if a company has posted new jobs.

## Usage

Create a virual environment using conda / virtualenv and install the necessary packages using 
`pip install -r requirements.txt`

Scrape, clean, parse using
`python run.py`

This will save a **job_lists.json**.

Compare two json files using the compare scripts provided

Try with models of your choice! DeepSeek-R1-Distill-Llama-8B gave me the best results :D. Requires about 6-8GB VRAM with 4 bit quantisation (enabled in the script)
