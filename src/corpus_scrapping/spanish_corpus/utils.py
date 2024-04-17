import unicodedata
from flatten_json import flatten
import spacy
import json
import pandas as pd
import unicodedata
from typing import List, Dict
from lxml import etree

nlp = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")


nlp.max_length = 1200000  # adjust as per your requirements
nlp_es.max_length = 1200000  # adjust as per your requirements

MIN_TOKENS = 5  # minimum number of tokens to index a passage


def unicode_parse(x: str) -> str:
    """
    Parses a str to convert it to unicode. Converts special characters, etc.
    """
    return unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf8')


def parse_and_chunk(document: str, chunk_size: int = 300, lang: str = 'en') -> List[str]:
    spacy_nlp = nlp_es if lang == 'es' else nlp


    doc = spacy_nlp(document)  # Move this line here, after the length check

    if len(doc) < chunk_size:
        # if passage fits in chunk size, return it immediately

        return [doc.text]

    passages = []
    sent_idx, sentences = 0, list(doc.sents)
    curr_passage, curr_passage_tokens = [], 0
    last_intro_paragraph = None

    while sent_idx < len(sentences):
        if curr_passage_tokens + len(sentences[sent_idx]) <= chunk_size:
            # if the full sentence still fits within the chunk size, continue
            curr_passage.append(sentences[sent_idx])
            curr_passage_tokens += len(sentences[sent_idx])
        else:
            if curr_passage:
                passages.append(" ".join([s.text for s in curr_passage]))
                if curr_passage[-1].text.strip().endswith(':'):
                    last_intro_paragraph = curr_passage[-1]
                else:
                    last_intro_paragraph = None

            curr_passage_tokens = len(sentences[sent_idx])
            curr_passage = [sentences[sent_idx]]
            if last_intro_paragraph is not None:
                curr_passage.insert(0, last_intro_paragraph)
                curr_passage_tokens += len(last_intro_paragraph)

        sent_idx += 1

    if curr_passage:
        passages.append(" ".join([s.text for s in curr_passage]))

    return passages


def chunk_paragraphs(paragraphs: List[str], lang: str) -> List[str]:
    """
    Takes paragraphs extracted from a webpage (i.e those enclosed in <p> tags) and chunks them up into
    chunk_size-length paragraphs
    """
    chunked_paragraphs = []
    for p in paragraphs:
        chunked = parse_and_chunk(document=p, lang=lang)
        chunked_paragraphs.extend(chunked)
    return chunked_paragraphs


# def format_list_items(list_items: List[str]) -> str:
#     if list_items:
#         list_items[-1] += "."
#         return "\\n- " + "\\n- ".join(list_items)
#     return ""


def parse_xml(xml_string: str, lang: str = 'en') -> List[str]:
    """
    Function to extract list content and paragraph content from documents.
    """
    root = etree.fromstring(xml_string)
    MAX_TOKENS = 300  # Added chunk size definition


    passages = []

    elements = root.xpath('//p | //list | //head')

    idx = 0

    while idx < len(elements):
        element = elements[idx]

        passage = ""

        # Check if element.text is not None and assign an empty string if it is
        element_text = element.text.strip() if element.text else ""
        if element.tag == 'p':
            passage += element.text.strip() if element.text else ""
            # Check if the paragraph ends with ':' and the next element is a list
            if passage.endswith(':') and idx + 1 < len(elements) and elements[idx + 1].tag == 'list':
                idx += 1  # move to the next element which is a list

                # Modify list items as per new requirements
                list_items = [
                    # Strip trailing period and convert
                    item.text.strip()
                    for item in elements[idx].xpath('.//item')
                    if item.text and item.text.strip()  # Ensure non-empty after stripping

                ]
                list_text = "\\n" + \
                    "\\n-".join(list_items) if list_items else ""

                # Add a period to the last list item if there are items
                if list_items:
                    list_items[-1] += "."

                # Join the modified list items with line breaks
                list_text = "\\n- " + \
                    "\\n- ".join(list_items) if list_items else ""
                passage += list_text

        # Header Handling
        if element.tag == 'head':
            header_text = element_text.rstrip(".,;:?!")  # strip punctuation
            # Check if header has ending punctuation (colon)
            if not header_text.endswith(':'):
                header_text += ":"  # Add colon if missing

            # Check if next element is a paragraph
            if idx + 1 < len(elements) and elements[idx + 1].tag == 'p':
                idx += 1  # Move to next element which is a paragraph
                # Combine the header and paragraph
                passage += " " + header_text + \
                    " " + \
                    elements[idx].text.strip() if elements[idx].text else ""

            # Check if next element is a list
            elif idx + 1 < len(elements) and elements[idx + 1].tag == 'list':
                idx += 1  # Move to next element which is a list
                list_items = [
                    item.text.strip()
                    for item in elements[idx].xpath('.//item/p')
                    if item.text and item.text.strip()  # Ensure non-empty after stripping
                ]
                list_text = "\\n-" + \
                    "\\n-".join(list_items) if list_items else ""

                passage += header_text + "\\n- " + "\\n- ".join(list_items)

        # List Handling (standalone lists)
        elif element.tag == 'list':
            list_items = [item.text.strip().rstrip(".")
                        for item in elements[idx].xpath('.//item/p')
                        if item.text and item.text.strip()  # Ensure non-empty after stripping
                        ]
            # Add a period to the last list item if there are items
            if list_items:
                list_text = "\\n- " + "\\n- ".join(list_items)
                list_items[-1] += "."
                # Add a period to the last list item if there are items
                if list_items:
                    list_items[-1] += "."

                passage += list_text

        # Add passage to passages if it's not empty
        if passage:
            passages.append(passage.strip())

        idx += 1

    return chunk_paragraphs(passages, lang)


def list_to_jsonlines_file(l: List[Dict], output_file: str):
    """
    Utility function to write a list of dicts to a jsonlines file.
    """
    with open(output_file, 'w', encoding='utf8') as outfile:
        for item in l:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')


def list_to_tsv(l: List[Dict], file: str):
    """
    Utility function to write a list of dicts to a tsv file.
    """
    pd.DataFrame(l).to_csv(file, sep='\t', index=False, encoding='utf-8')
