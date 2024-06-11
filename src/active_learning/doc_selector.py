import gzip
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import sparse
import pandas as pd

class DocSelector(object):
    def __init__(
        self,
        path_model: Path,
        path_source: Path,
        polylingual: bool = False,
        lang: str = None,
        logger: logging.Logger = None
    ) -> None:

        self._logger = logger if logger else logging.getLogger(__name__)
        self.output_lang = self._load_model_info(
            path_model, path_source, polylingual, lang)

        # Mapping of identifiers to method names
        self._method_mapping = {
            1: 'identify_bad_documents_v1',
            2: 'identify_bad_documents_v1_norm',
            3: 'identify_bad_documents_v2',
            4: 'identify_bad_documents_v2_norm',
            5: 'identify_bad_documents_v3',
            6: 'identify_bad_documents_v3_norm'
        }

    def invoke_method_by_id(self, identifier: int):

        self._logger.info(
            f"-- -- Using method with identifier: {identifier} - {self._method_mapping.get(identifier)} for 'bad' documents metric... ")

        # Get the method name from the mapping
        method_name = self._method_mapping.get(identifier)

        if method_name is None:
            raise ValueError(f"Invalid identifier: {identifier}")

        # Get the method using getattr and invoke it
        method = getattr(self, method_name)

        return method(self.output_lang)

    def _load_model_info(
        self,
        path_model: Path,
        path_source: Path,
        polylingual: bool = False,
        lang: str = None
    ):
        if polylingual:
            pass
        else:
            ###############
            # Corpus      #
            ###############

            #  Read the source data
            self._logger.info(f"-- -- Loading source data from {path_source}... ")
            print(f"-- -- Loading source data from {path_source}... ")
            raw_lang = pd.read_parquet(path_source)
            path_corpus_lang = path_model / "train_data" / f"corpus_{lang}.txt"
            with path_corpus_lang.open("r", encoding="utf-8") as f:
                lines = [line for line in f.readlines()]
                corpus_lang = [line.rsplit(
                    " 0 ")[1].strip().split() for line in lines]

                ids = [line.split(" 0 ")[0] for line in lines]
                df_lang = pd.DataFrame(
                    {"lemmas": [" ".join(doc) for doc in corpus_lang]})
                df_lang["doc_id"] = ids
                df_lang["len"] = df_lang['lemmas'].apply(
                    lambda x: len(x.split()))
                df_lang["id_top"] = range(len(df_lang))

            df_lang_raw = df_lang.merge(raw_lang, how="inner", on="doc_id")[
                ["doc_id", "id_top", "id_preproc", "lemmas_x", "text", "len"]]
            self._logger.info(f"-- -- Loaded {len(df_lang_raw)} documents... ")
            print(f"-- -- Loaded {len(df_lang_raw)} documents... ")

            ###############
            # Topic state #
            ###############
            self._logger.info(f"-- -- Loading topic state from {path_model / f'mallet_output/{lang}/topic-state.gz'}... ")
            with gzip.open((path_model / f"mallet_output/{lang}" / "topic-state.gz")) as fin:
                topic_state_df = pd.read_csv(
                    fin, sep='\s+',
                    names=['docid', 'NA3', 'wd_idx_doc', 'wd_vocab', 'word', 'tpc'], header=None, skiprows=3)

                # Get assignments
                z = topic_state_df.copy().groupby(['docid'])[
                    'tpc'].apply(list).reset_index(name='new')
                z = z.new.values.tolist()

                # Get IDs of words in documents
                documents = topic_state_df.copy().groupby(
                    ['docid'])['wd_idx_doc'].apply(list).reset_index(name='new')
                documents = documents.new.values.tolist()

                # Get actual content of the documents
                documents_texts = topic_state_df.copy().groupby(
                    ['docid'])['word'].apply(list).reset_index(name='new')
                documents_texts = documents_texts.new.values.tolist()
            self._logger.info(f"-- -- Loaded {len(z)} documents... ")
            print("z", z[0:100], len(z))
            print("documents", documents[0], len(documents))
            print("documents_texts", documents_texts[0], len(documents_texts))
            
            
            ######################################
            # Keep only documents after training
            ######################################
            # Filter out non-string elements and join strings in sublists
            print(f"-- -- Filtering out non-string elements and joining strings in sublists... ")
            aux = [" ".join([str(item) for item in sublist]) for sublist in documents_texts]
            df_aux = pd.DataFrame(aux, columns=["lemmas_x"])
            kept_docs_tm = topic_state_df.docid.unique().tolist()
            df_aux["id_top"] = kept_docs_tm
            df_ = df_aux.merge(df_lang_raw, how="inner", on="id_top")
            print(f"-- -- Filtered {len(df_)} documents... ")
            #import pdb; pdb.set_trace()
            ################
            # Thetas       #
            ################
            self._logger.info(f"-- -- Loading thetas from {path_model / f'mallet_output/{lang}/thetas.npz'}... ")
            thetas = sparse.load_npz(
                (path_model / f"mallet_output/{lang}" / "thetas.npz")).toarray()
            print("thetas", thetas.shape)
            print(f"-- -- Filtering thetas... ")
            thetas = thetas[kept_docs_tm]
            print("thetas", thetas.shape)

            ################
            # Betas        #
            ################
            self._logger.info(f"-- -- Loading betas from {path_model / f'mallet_output/{lang}/betas.npy'}... ")
            betas = np.load(
                (path_model / f"mallet_output/{lang}" / "betas.npy"))
            print("betas", betas.shape)

            ################
            # Vocab        #
            ################
            self._logger.info(f"-- -- Loading vocab from {path_model / f'mallet_output/{lang}/vocab_freq.txt'}... ")
            vocab_w2id = {}
            vocab_id2w = {}
            with open((path_model / f"mallet_output/{lang}" / "vocab_freq.txt")) as file:
                for i, line in enumerate(file):
                    # Strip leading and trailing whitespace
                    stripped_line = line.strip()
                    # Split the line into words and numbers
                    parts = stripped_line.split()
                    if parts:
                        # Get the word (first part)
                        wd = parts[0]
                        # Populate the dictionaries
                        vocab_w2id[wd] = i
                        vocab_id2w[str(i)] = wd

            ################
            # Topic keys   #
            ################
            self._logger.info(f"-- -- Loading topic keys from {path_model / f'mallet_output/{lang}/topickeys.txt'}... ")
            keys = []
            with open((path_model / f"mallet_output/{lang}" / "topickeys.txt"), 'r') as file:
                for line in file:
                    # Strip leading and trailing whitespace
                    stripped_line = line.strip()
                    # Split the line into parts and ignore the first two parts (number and float)
                    parts = stripped_line.split(maxsplit=2)
                    if len(parts) > 2:
                        text_part = parts[2]
                        keys.append(text_part)

            output_lang = {
                "z": z,
                "documents": documents,
                "documents_texts": documents_texts,
                "thetas": thetas,
                "betas": betas,
                "vocab_w2id": vocab_w2id,
                "vocab_id2w": vocab_id2w,
                "keys": keys,
                "df": df_,
                "corpus": corpus_lang###
            }

        return output_lang

    ###############
    # Approach 1: #
    ###############
    # Identify bad documents using actual topic assignments and log-probabilities
    def identify_bad_documents_v1(self, output_lang: Dict):
        betas = output_lang["betas"].copy()
        z = output_lang["z"]
        documents = output_lang["documents"]

        epsilon = 1e-10
        betas += epsilon

        # Calculate log-probabilities for numerical stability
        log_betas = np.log(betas)

        D = len(documents)
        doc_log_probs = np.zeros(D)

        for d in range(D):
            doc = documents[d]
            topic_assignments = z[d]
            doc_log_prob = 0
            for i, word in enumerate(doc):
                topic = topic_assignments[i]
                doc_log_prob += log_betas[topic, word]
            doc_log_probs[d] = doc_log_prob

        # Rank documents by their log-probabilities (lower is worse)
        bad_document_indices = np.argsort(doc_log_probs)
        return bad_document_indices, doc_log_probs[bad_document_indices]

    def identify_bad_documents_v1_norm(self, output_lang: Dict):
        betas = output_lang["betas"].copy()
        z = output_lang["z"]
        documents = output_lang["documents"]

        epsilon = 1e-10
        betas += epsilon

        # Calculate log-probabilities for numerical stability
        log_betas = np.log(betas)

        D = len(documents)
        doc_log_probs = np.zeros(D)

        for d in range(D):
            doc = documents[d]
            topic_assignments = z[d]
            doc_log_prob = 0
            for i, word in enumerate(doc):
                topic = topic_assignments[i]
                doc_log_prob += log_betas[topic, word]
            # Normalize the log-probability by the length of the document
            doc_log_probs[d] = doc_log_prob / len(doc)

        # Rank documents by their normalized log-probabilities (lower is worse)
        bad_document_indices = np.argsort(doc_log_probs)
        return bad_document_indices, doc_log_probs[bad_document_indices]

    ###############
    # Approach 2: #
    ###############
    # Identify bad documents by summing weights assuming all words from one topic
    def identify_bad_documents_v2(self, output_lang: Dict):
        thetas = output_lang["thetas"].copy()
        betas = output_lang["betas"].copy()
        documents_texts = output_lang["documents_texts"]
        vocab_w2id = output_lang["vocab_w2id"]

        D = len(thetas)
        K = len(betas)
        S3 = np.zeros((D, K))

        for doc in range(D):
            for topic in range(K):
                wd_ids = [vocab_w2id[word]
                          for word in documents_texts[doc] if word in vocab_w2id]
                S3[doc, topic] = np.sum(betas[topic, wd_ids])

        return S3

    def identify_bad_documents_v2_norm(self, output_lang: Dict):
        thetas = output_lang["thetas"].copy()
        betas = output_lang["betas"].copy()
        documents_texts = output_lang["documents_texts"]
        vocab_w2id = output_lang["vocab_w2id"]

        D = len(thetas)
        K = len(betas)
        S3 = np.zeros((D, K))
        for doc in range(D):
            for topic in range(K):
                wd_ids = [vocab_w2id[word]
                        for word in documents_texts[doc] if word in vocab_w2id]
                try:
                    S3[doc, topic] = (1/len(wd_ids)) * \
                        np.sum(betas[topic, wd_ids])
                except Exception as e:
                    #print(e)
                    continue
        return S3

    ###############
    # Approach 3: #
    ###############
    # Identify bad documents by summing weights assuming of words assigned to the given topic
    def identify_bad_documents_v3(self, output_lang: Dict):
        thetas = output_lang["thetas"].copy()
        betas = output_lang["betas"].copy()
        z = output_lang["z"]
        documents = output_lang["documents"]
        documents_texts = output_lang["documents_texts"]
        vocab_w2id = output_lang["vocab_w2id"]

        D = len(thetas)
        K = len(betas)
        S3 = np.zeros((D, K))

        for doc in range(D):
            for topic in range(K):
                try:
                    wd_ids = [i for i, word in zip(
                        documents[doc], documents_texts[doc]) if word in vocab_w2id and z[doc][i] == topic]
                    S3[doc, topic] = np.sum(betas[topic, wd_ids])
                except Exception as e:
                    continue
        return S3

    def identify_bad_documents_v3_norm(self, output_lang: Dict):
        thetas = output_lang["thetas"].copy()
        betas = output_lang["betas"].copy()
        z = output_lang["z"]
        documents = output_lang["documents"]
        documents_texts = output_lang["documents_texts"]
        vocab_w2id = output_lang["vocab_w2id"]
        D = len(thetas)
        K = len(betas)
        S3 = np.zeros((D, K))

        for doc in range(D):
            for topic in range(K):
                try:
                    wd_ids = [i for i, word in zip(
                        documents[doc], documents_texts[doc]) if word in vocab_w2id and z[doc][i] == topic]
                    S3[doc, topic] = (1/len(wd_ids)) * \
                        np.sum(betas[topic, wd_ids])
                except Exception as e:
                    continue
        return S3
