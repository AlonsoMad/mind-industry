#!/usr/bin/python
import json
from subprocess import check_output
import os
import time
import pathlib
import re

####################
# 1. Data crawling #
####################
print(f"{'*'*20} Data crawling {'*'*20}")
# Get current directory (where the script is located)
current_dir = pathlib.Path(os.getcwd())
path_scrapper = current_dir.parent / "corpus_scraping" / "run.py"
kb_sources_file = "kb_spanish_sources.json"

# Parameters for scrapper
scrape_rule = '\"strict\"'
scrape_save = scrape_rule[1:-1]
depth_limit = 3
kb_sources_path = '\"' + (current_dir / kb_sources_file).as_posix() + '\"'

cmd = "python3 " + path_scrapper.as_posix() + \
    " --scrape_rule=%s --depth_limit=%s --kb_sources_path=%s"
cmd = cmd % (scrape_rule, depth_limit, kb_sources_path)

time_start = time.time()
try:
    print(f"-- -- Running scrapper...")
    check_output(args=cmd, shell=True)
except Exception as e:
    print(f"-- -- {e}")
    print(f"-- -- Error running scrapper")
    exit(1)
time_end = time.time() - time_start
print(f"-- -- Scrapper finished in {time_end} seconds")


####################
# 2. Fix jsons     #
####################
print(f"{'*'*20} Fix jsons {'*'*20}")
source_data_path = current_dir / scrape_save
edited_data_path = current_dir / "es_database" / "scraped"
processed_data_path = current_dir / "es_database" / "processed"
edited_data_path.mkdir(parents=True, exist_ok=True)
processed_data_path.mkdir(parents=True, exist_ok=True)


def extract_blocks_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Use regular expression to find all matches of {} blocks
    pattern = r'\{([^}]*)\}'
    matches = re.findall(pattern, content)

    # Extract the content inside each {} block and store in a list
    extracted_content = [match.strip() for match in matches]

    return extracted_content


for file in source_data_path.iterdir():
    if file.suffix == '.json':
        print(f"-- -- Fixing {file.name}")

        # Store the extracted content in a list
        extracted_content_list = extract_blocks_from_file(file)

        all_elements = []
        for el in extracted_content_list:
            # Split the content of each {} block by ', "'
            elements = el.split(', "')
            elements_dict = {}
            for e in elements:
                key = e.split('": ')[0].strip('"')
                value = e.split('": ')[1].strip('"').replace('\\"', '"')
                elements_dict[key] = value
            all_elements.append(elements_dict)
        path_save = edited_data_path / file.name
        with open(path_save, "w") as json_file:
            json.dump(all_elements, json_file)


############################
# 3. Prepare scrapped docs #
############################
print(f"{'*'*20} Prepare scrapped docs {'*'*20}")
path_prepare_docs = current_dir / "prepared_scraped_docs.py"

name = "es"
language = name
version = "1.0"

cmd = "python3 " + path_prepare_docs.as_posix() + \
    " --scraped_documents_dir=%s --out_dir=%s --name=%s --language=%s --version=%s"
cmd = cmd % (edited_data_path.as_posix(), processed_data_path.as_posix(), name, language, version)

time_start = time.time()
try:
    print(f"-- -- Running prepare docs...")
    check_output(args=cmd, shell=True)
except Exception as e:
    print(e)
    print(f"-- -- Error running prepare docs")
    exit(1)
time_end = time.time() - time_start
print(f"-- -- Prepare docs finished in {time_end} seconds")