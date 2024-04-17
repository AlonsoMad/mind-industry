import os
import json
import argparse
from typing import List
import logging
import tqdm
import spacy
import numpy as np
import pandas as pd
from lxml import etree
import concurrent.futures


import langdetect
from utils import parse_and_chunk, parse_xml, list_to_jsonlines_file

# Configure the logging settings
logging.basicConfig(
    filename='processing3.log',  # Specify the log file name
    level=logging.INFO,         # Set the logging level to INFO or your preferred level
    # Define the log message format
    format='%(asctime)s - %(levelname)s - %(message)s'
)
MIN_TOKENS = 10  # minimum number of tokens to index a passage
"""
Switch to Spanish when working with Spanish corpus 
"""
#ALLOWLISTED_LANGUAGES = ['en']
ALLOWLISTED_LANGUAGES = ['es']


def manual_kb_to_passages(manual_kb: str = '../knowledge_base/knowledge_base.json') -> pd.DataFrame:
    """
    Utility function to convert human-written KB answers to passages.
    Designed to mimic the format of scraped sources.
    """
    kb = []
    for i in json.load(open('../knowledge_base/knowledge_base.json', 'r'))['questions']:
        for c in parse_and_chunk(i['answer']):
            kb.append({
                # hacky: we want to include all manual written answers no matter what
                'has_trigger_word': True,
                'url': i['data_source'],
                'document_text': c,
                'title': None,
            })
    return pd.DataFrame(kb)


class ScrapedDocumentCompiler:

    chunksize = 10000
    """
    Compiles per-parent source json lines file and manual KB passages into a single 
    json lines file to be ingested by the Corpus() class.
    """

    def __init__(self,
                 name: str,
                 version: str,
                 sources_dir: str,
                 out_dir: str,
                 filter_documents_no_trigger_words: bool = False,
                 language: str = 'en'
                 ) -> None:
        self.name = name
        self.version = version
        self.out_dir = out_dir
        self.sources_dir = sources_dir
        self.language = language
        self.documents = self.process_documents_per_source(
            sources_dir, filter_documents_no_trigger_words)

    def process_documents_per_source(self, sources_dir: str, filter_documents_no_trigger_words: bool):
        """
        Process all scraped documents in each source json lines file into a single dataframe.
        """
        corpus_documents = []
        for file in sorted(os.listdir(self.sources_dir)):
            # process all source-domain specific files
            print(f"ENDING: {file}")
            if file.endswith(".json"):
                logging.info(f"Processing file: {file}, document: {file}")

                print(file)
                try:
                    #import pdb; pdb.set_trace()
                    #for chunk in pd.read_json(os.path.join(self.sources_dir, file),lines=True, chunksize=self.chunksize):#lines=True
                    corpus_documents.append(
                        pd.read_json(os.path.join(self.sources_dir, file))
                    )

                except ValueError as e:
                    print(f"Failed to read {file}: {e}")
        # corpus_documents.append(manual_kb_to_passages())
        documents = pd.concat(corpus_documents).reset_index(drop=True)

        # if filter_documents_no_trigger_words:
        #     return documents[documents['has_trigger_word'] == True].reset_index(drop=True) #admit despite no trigger word 

        print('Finished processing...total %d docs' % (len(documents)))
        return documents

    def compile_document_based_corpus(self) -> None:
        """
        Compile a document-based corpus from the scraped documents.
        (Not chunked into passages)
        """
        outfile_name = f'{self.name}_v{self.version}_compiled_documents.jsonl'

        text_documents = []

        for idx, (row, parent_doc) in enumerate(tqdm.tqdm(self.documents.iterrows(), total=len(self.documents))):
            # Extract the progress string
            progress_string = f"{idx + 1}/{len(self.documents)}"

            # Log the progress string
            logging.info(f"Processing record {progress_string}")

            # Debugging: Print the 'title' value to inspect its content
            print(
                f"Record {progress_string} - 'title' value: {parent_doc['title']}")

            # Add the document information to the text_documents list
            text_document = {
                'document_id': str(row),
                'url': parent_doc['url'],
                'has_trigger_word': parent_doc['has_trigger_word'],
                'title': u"" if not parent_doc['title'] else str(parent_doc['title']).strip(),
                'contents': parent_doc['document_text'].replace('\n', ' ').strip()
            }
            text_documents.append(text_document)

        # After the loop, save text_documents to the JSON file
        list_to_jsonlines_file(
            text_documents, os.path.join(self.out_dir, outfile_name))

        return text_documents

    # Noticed duplicated passages. Added code to remove those dups

    def compile_passage_based_corpus(self, batch_size: int = 50000) -> None:
        """
        Compile a passage-based corpus from the scraped documents.
        Passages are between MIN_TOKENS (10) and 100 tokens long.
        """
        outfile_name = f'{self.name}_v{self.version}_compiled_passages.jsonl'
        num_records = self.documents.shape[0]
        print(f' num_records {num_records}')

        # Initialize an empty passage_corpus and seen_passages set for this batch
        passage_corpus = []
        # seen_passages = set()

        # Open the JSON file for writing
        with open(os.path.join(self.out_dir, outfile_name), 'a', encoding='utf-8') as file:

            # Processing in Batches
            for i in range(0, num_records, batch_size):
                print(f"Processing batch {i // batch_size + 1}...")
                batch_data = self.documents.iloc[i:i+batch_size]

                for row, parent_doc in tqdm.tqdm(batch_data.iterrows(), total=batch_data.shape[0]):
                    if not parent_doc['document_xml'] or not isinstance(parent_doc['document_xml'], str):
                        logging.info(f"Skipped document {row} due to no XML")
                        continue
                    
                    #import pdb; pdb.set_trace()
                    
                    try:
                        root = etree.fromstring(parent_doc['document_xml'])
                        passages = parse_xml(etree.tostring(root))
                        print("XML parsing successful!")
                    except etree.XMLSyntaxError as e:
                        print(f"XML parsing error: {e}")
                        
                    if not passages:
                        logging.info(f"Skipped document {row} due to no passages")
                        
                    for num, passage in enumerate(passages):
                        not_short = len(passage.split()) >= MIN_TOKENS
                        try:
                            is_allowed_language = langdetect.detect(passage) in ALLOWLISTED_LANGUAGES  # HM UPDATES
                        except langdetect.LangDetectException:
                            logging.info(f"Langdetect exception in document {row} passage {num}")
                            is_allowed_language = True

                        if not_short and is_allowed_language:
                            # Check if the 'title' attribute is an integer and handle it
                            title = u"" if not parent_doc['title'] else parent_doc['title']
                            if isinstance(title, int):
                                title = str(title)

                            # Calculate a hash for the current passage
                            # passage_hash = hash(passage)
                            # if row == 2381:
                            #     print(passage_hash in seen_passages)
                            # if passage_hash not in seen_passages:  # Check if this passage hash has not been seen before HM UPDATES
                            passage_dict = {
                                'passage_id': "%s-%s" % (row, num),
                                'passage': passage.replace('\n', ' ').strip(),
                                'title': u"" if not title else title.replace('\n', ' ').strip(),
                                'url': parent_doc.url,
                            }
                            passage_corpus.append(passage_dict)
                            # seen_passages.add(passage_hash)

                            # Write the passage to the JSON file immediately
                            json.dump(passage_dict, file, ensure_ascii=False)
                            file.write("\n")
                        else:
                            logging.info(
                                f"Skipped passage {num} in document {row} due to length or language: {passage}")

                # Clear the batch-specific data to save memory
                passage_corpus = []
                # seen_passages = set()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scraped_documents_dir', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--version', type=str, default='')
    parser.add_argument('--language', type=str, default='es')
    parser.add_argument(
        '--filter_documents_no_trigger_words', action='store_true')

    args = parser.parse_args()

    compiler = ScrapedDocumentCompiler(
        args.name,
        args.version,
        args.scraped_documents_dir,
        args.out_dir,
        args.filter_documents_no_trigger_words ,
        args.language
    )

    compiler.compile_document_based_corpus()
    compiler.compile_passage_based_corpus()
