import pandas as pd
import torch
from transformers import pipeline
from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize



df = pd.read_parquet("/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/corpus_rosie/preproc/en_0.0001.parquet")[0:2]

# def translate(text):
#     pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-13B-v0.1", device_map="auto")# torch_dtype=torch.bfloat16, 
#     # We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
#     messages = [
#         {"role": "user", "content": f"Translate the following text from English into Spanish.\n{text}.\Spanish:"},
#     ]
#     prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     outputs = pipe(prompt, max_new_tokens=256, do_sample=False)

#     return outputs[0]["generated_text"]

def translate(text):
    
    target_language = "es"
    model_name = f'Helsinki-NLP/opus-mt-en-{target_language}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    def chunk_text(text, chunk_size):
        # Check if chunk_size is valid
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0")

        # Initialize an empty list to store the chunks
        chunks = []

        # Loop through the text and create chunks of the specified size
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)

        return chunks

    if len(text) > tokenizer.model_max_length:
        texts_splits = chunk_text(text, tokenizer.model_max_length)
    
    def translate_sentence(sentence):
        inputs = tokenizer.encode(sentence, return_tensors="pt")
        outputs = model.generate(inputs, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    
    translation = ""
    for sentence in texts_splits:
        translation += translate_sentence(sentence) + " "
    
    translation = translation.strip()
        
    return translation
    
#df["translated"] = df["raw_text"].apply(translate)

def translate_corpus(
    df: pd.DataFrame,
    source_language: str = "en",
    target_language: str = "es"
) -> pd.DataFrame:
    """
    Translate a corpus from a source language to a target language.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the corpus to translate.
    source_language : str
        Source language of the corpus. Default is "en".
    target_language : str
        Target language of the corpus. Default is "es".

    Returns
    -------
    pd.DataFrame
        DataFrame with the translated corpus. The columns are the same as the input DataFrame, plus a new column "raw_text_tr" with the translated text.
    """

    # Function to translate a sentence using the provided tokenizer and model
    def translate_sentence(tokenizer, model, sentence):
        inputs = tokenizer.encode(sentence, return_tensors="pt")
        outputs = model.generate(inputs, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    # Load the translation model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Function to translate a text, handling chunking if necessary
    def translate(text):
        if len(text) > tokenizer.model_max_length:
            texts_splits = [text[i:i + tokenizer.model_max_length] for i in range(0, len(text), tokenizer.model_max_length)]
        else:
            texts_splits = [text]
        
        translations = [translate_sentence(tokenizer, model, sentence) for sentence in texts_splits]
        return ' '.join(translations)

    # Apply translation function to each text in the DataFrame
    df["raw_text_tr"] = df["raw_text"].apply(translate)
    return df

df = translate_corpus(df)
import pdb; pdb.set_trace()