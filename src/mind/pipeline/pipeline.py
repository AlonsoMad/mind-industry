from pathlib import Path
import re
from typing import List, Union
from dotenv import dotenv_values

import pandas as pd
import torch
from colorama import Fore, Style
from mind.pipeline.corpus import Corpus
from mind.pipeline.retriever import IndexRetriever
from mind.pipeline.utils import extend_to_full_sentence
from mind.prompter.prompter import Prompter
from mind.utils.utils import init_logger, load_prompt, load_yaml_config_file
from sentence_transformers import SentenceTransformer  # type: ignore
from tqdm import tqdm  # type: ignore
from collections import defaultdict

import re
import unicodedata


class MIND:
    """
    MIND pipeline class.

    =========================                
    All components are build based on the Prompter class. Each component has its own template and is responsible for generating the input for the next component.

    We apply additional non-LLM based filters:
    - Filter out yes/no questions
    - Check that the generated answer entails the original passage (if enabled)
    - If the target answer contains "I cannot answer the question", we directly label it as NOT_ENOUGH_INFO without checking for contradiction
    """

    def __init__(
        self,
        llm_model: str,
        llm_server: str = None,
        source_corpus: Union[Corpus, dict] = None,
        target_corpus: Union[Corpus, dict] = None,
        retrieval_method: str = "TB-ENN",
        multilingual: bool = True,
        lang: str = "en",
        config_path: Path = Path("config/config.yaml"),
        logger=None,
        dry_run: bool = False,
        do_check_entailement: bool = False,
        env_path=None,
    ):
        self._logger = logger if logger else init_logger(config_path, __name__)

        self.dry_run = dry_run
        self.retrieval_method = retrieval_method

        self.config = load_yaml_config_file(config_path, "mind", self._logger)
        self._embedding_model = self.config.get("embedding_models", {}).get("multilingual").get("model") if multilingual else self.config.get(
            "embedding_models", {}).get("monolingual").get(lang).get("model")
        self._do_norm = self.config.get("embedding_models", {}).get("multilingual").get("do_norm") if multilingual else self.config.get(
            "embedding_models", {}).get("monolingual").get(lang).get("do_norm")

        self.cannot_answer_dft = self.config.get(
            "cannot_answer_dft", "I cannot answer the question given the context.")
        self.cannot_answer_personal = self.config.get(
            "cannot_answer_personal", "I cannot answer the question since the context only contains personal opinions.")

        env_path = env_path or self.config.get(
            "llm", {}).get("gpt", {}).get("env_path")

        try:
            open_api_key = dotenv_values(env_path).get("OPEN_API_KEY", None)
        except Exception as e:
            self._logger.error(f"Failed to load environment variables: {e}")

        self._prompter = Prompter(
            model_type=llm_model,
            llm_server=llm_server,
            config_path=config_path,
            openai_key=open_api_key
        )

        self._prompter_answer = Prompter(
            model_type=llm_model,
            llm_server=llm_server,
            config_path=config_path,
            openai_key=open_api_key
        )
        self.prompts = {}
        for name in ["question_generation", "subquery_generation", "answer_generation", "contradiction_checking", "relevance_checking"]:
            path = self.config.get("prompts", {}).get(name)
            if path is None:
                raise ValueError(f"Missing prompt path for: {name}")
            self.prompts[name] = load_prompt(path)

        # NLI components
        self._do_check_entailment = do_check_entailement
        if self._do_check_entailment:

            self._nli_model_name = self.config.get(
                "nli_model_name", "potsawee/deberta-v3-large-mnli")
            try:
                from transformers import (  # type: ignore
                    AutoModelForSequenceClassification, AutoTokenizer)
                self._nli_tokenizer = AutoTokenizer.from_pretrained(
                    self._nli_model_name, use_fast=False)
                self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                    self._nli_model_name)
                self._logger.info(f"NLI model loaded: {self._nli_model_name}")
            except Exception as e:
                self._logger.error(f"Failed to load NLI model/tokenizer: {e}")
                self._nli_tokenizer, self._nli_model = None, None

        if source_corpus:
            self.source_corpus = self._init_corpus(
                source_corpus, is_target=False)
        if target_corpus:
            self.target_corpus = self._init_corpus(
                target_corpus, is_target=True)

        if self.dry_run:
            self._logger.warning(
                "Dry run mode is ON — no LLM calls will be made.")

        self.results = []
        self.discarded = []

        # keep a unique identifier for the questions generated per topic
        self.questions_id = defaultdict(set)

    def _init_corpus(
        self,
        corpus: Union[Corpus, dict],
        is_target: bool = False
    ) -> Corpus:

        if isinstance(corpus, Corpus):
            return corpus

        required_keys = {"corpus_path", "id_col",
                         "passage_col", "full_doc_col"}

        if not required_keys.issubset(corpus.keys()):
            raise ValueError(
                f"Missing keys in corpus config dict: {required_keys - corpus.keys()}")

        corpus_obj = Corpus.from_parquet_and_thetas(
            path_parquet=corpus["corpus_path"],
            path_thetas=Path(corpus["thetas_path"]) if corpus.get(
                "thetas_path") else None,
            id_col=corpus["id_col"],
            passage_col=corpus["passage_col"],
            full_doc_col=corpus["full_doc_col"],
            row_top_k=corpus.get("row_top_k", "top_k"),
            language_filter=corpus.get("language_filter", None),
            logger=self._logger,
            load_thetas=corpus.get("load_thetas", True),
            filter_ids=corpus.get("filter_ids", None)
        )

        self._logger.info(
            f"Corpus {corpus['corpus_path']} loaded with {len(corpus_obj.df)} documents.")

        if is_target:
            retriever = IndexRetriever(
                model=SentenceTransformer(self._embedding_model),
                logger=self._logger,
                top_k=self.config.get("mind", {}).get("top_k", 10),
                batch_size=self.config.get("mind", {}).get("batch_size", 32),
                min_clusters=self.config.get(
                    "mind", {}).get("min_clusters", 8),
                do_weighting=self.config.get(
                    "mind", {}).get("do_weighting", True),
                nprobe_fixed=self.config.get(
                    "mind", {}).get("nprobe_fixed", False),

            )
            retriever.build_or_load_index(
                source_path=corpus["corpus_path"],
                thetas_path=corpus["thetas_path"],
                save_path_parent=corpus["index_path"],
                method=corpus.get("method", "TB-ENN"),
                col_to_index=corpus["passage_col"],
                col_id=corpus["id_col"],
                lang=corpus.get("language_filter", None),
            )
            corpus_obj.retriever = retriever

        return corpus_obj

    def run_pipeline(self, topics, sample_size=None, previous_check=None, path_save="mind_results.parquet"):

        #  ensure path_save directory exists
        Path(path_save).mkdir(parents=True, exist_ok=True)

        for topic in topics:
            self._process_topic(
                topic, path_save, previous_check=previous_check, sample_size=sample_size)

    def _process_topic(self, topic, path_save, previous_check=None, sample_size=None):
        for chunk in tqdm(self.source_corpus.chunks_with_topic(
            topic_id=topic,
            sample_size=sample_size,
            previous_check=previous_check
        ), desc=f"Topic {topic}"):
            self._process_chunk(chunk, topic, path_save)

    def _process_chunk(self, chunk, topic, path_save):
        self._logger.info(f"Processing chunk {chunk.id} for topic {topic}")

        questions = chunk.metadata.get("questions")

        if questions:
            self._logger.info(
                f"Using preloaded questions from chunk {chunk.id}")
        else:
            questions, _ = self._generate_questions(chunk)
        if questions == []:
            print(
                f"{Fore.RED}No questions generated for chunk {chunk.id}{Style.RESET_ALL}")
            return
        self._logger.info(f"Generated questions: {questions}\n")
        for question in questions:
            self._process_question(question, chunk, topic, path_save)

    def _process_question(self, question, source_chunk, topic, path_save):
        # generate answer in source language
        a_s = None
        answers = source_chunk.metadata.get("answers")
        if answers and isinstance(answers, dict):
            a_s = answers.get(question)

        if not a_s:
            a_s, _ = self._generate_answer(question, source_chunk)
            self._logger.info(f"Generated original answer: {a_s}\n")

            # check that the answer entails the original chunk
            # if not, discard the question
            if self._do_check_entailment:
                _, _, entails = self._check_entailement(a_s, source_chunk.text)
                if not entails:
                    print(f"{Fore.RED}Discarding question '{question}' since the answer does not entail the original passage.{Style.RESET_ALL}\n ANSWER: {a_s}\nPASSAGE: {source_chunk.text}\n")
                    # self._logger.info(f"Discarding question '{question}' since the answer does not entail the original passage.\n ANSWER: {a_s}\nPASSAGE: {source_chunk.text}\n")
                    self.discarded.append({
                        "topic": topic,
                        "question": question,
                        "source_chunk": source_chunk.text,
                        "a_s": a_s,
                        "reason": "Answer does not entail the original passage"
                    })
                    return
        else:
            self._logger.info(
                f"Using preloaded answer from chunk {source_chunk.id}: {a_s}\n")

        # generate subqueries
        subqueries, _ = self._generate_subqueries(question, source_chunk)
        self._logger.info(f"Generated subqueries: {subqueries}\n")

        # generate answer in target language for each subquery and target chunk
        all_target_chunks = []
        for subquery in subqueries:
            target_chunks = self.target_corpus.retrieve_relevant_chunks(
                query=subquery, theta_query=source_chunk.metadata["top_k"])
            all_target_chunks.extend(target_chunks)
        # remove duplicates by chunk.id
        len_target_chunks = len(all_target_chunks)
        unique_target_chunks = {}
        for tc in all_target_chunks:
            if tc.id not in unique_target_chunks:
                unique_target_chunks[tc.id] = tc
        all_target_chunks = list(unique_target_chunks.values())
        self._logger.info(
            f"Retrieved {len_target_chunks} target chunks, {len(all_target_chunks)} unique.")

        for target_chunk in all_target_chunks:
            self._evaluate_pair(question, a_s, source_chunk,
                                target_chunk, topic, subquery, path_save)

    def _filter_bad_questions(questions: List[str]) -> List[str]:
        """Remove questions that are not well-formed or relevant.

        Parameters
        ----------
        questions : list[str]
            List of questions to filter.

        Returns
        -------
        list[str]
            Filtered list of questions.

        Filters applied
        ---------------
        1) Keep only yes/no style questions, trimming any preamble.
        2) Structural filters: must end with '?', have at least 3 words,
        and not start with follow-up openers like "and", "but", "or", "so".
        3) Remove questions that reference studies, reports, documents, sections, or
        sample-specific phrases, including participle phrasing like
        "results indicated..." or "the report summarized...".
        """

        # --- Normalization (expanded to your requested version) ---
        def _normalize(s: str) -> str:
            s = unicodedata.normalize("NFKC", s)
            s = re.sub(r"\s+", " ", s).strip()
            s = re.sub(r"^[\s\-\–\—\•\"'“”‘’«»]+", "", s)
            return s

        # yes/no auxiliary verbs
        _aux = (
            "is", "are", "am", "was", "were",
            "do", "does", "did",
            "has", "have", "had",
            "can", "could",
            "will", "would",
            "shall", "should",
            "may", "might",
            "must"
        )
        _aux_re = re.compile(rf"^\W*(?:{'|'.join(_aux)})\b", re.IGNORECASE)

        # follow-up openers
        _followup_openers = ("and", "but", "or", "so")

        # study/doc/sample-like phrases
        _DOC = (
            r"(?:study|survey|report|document|guidance|paper|article|memo|white\s*paper|brief|"
            r"dataset|discussion|section|appendix|table|figure|results?)"
        )
        _VERB = (
            r"(?:include|mention|provide|state|say|note|discuss|address|cover|contain|list|"
            r"describe|reference|present|report|indicate|summarize|focus)"
        )
        _PART = (
            r"(?:included|mentioned|provided|stated|noted|discussed|addressed|covered|contained|"
            r"listed|described|referenced|presented|reported|indicated|summarized|focused)"
        )

        _P1 = rf"\baccording to (?:the |this |that )?(?:results?|{_DOC})\b"
        _P2 = rf"\b(?:do|does|did|has|have|had)\s+(?:(?:the|this|that|these|those)\s+)?{_DOC}\s+{_VERB}\b"
        _P3 = rf"\b(?:in|within|from)\s+(?:(?:the|this|that)\s+)?{_DOC}\b"
        _P4 = rf"\b(?:selected|surveyed|polled|sampled|enrolled)\s+(?:respondents?|participants?|subjects?)\b"
        _P5 = rf"\bdid\s+(?:the\s+)?(?:study|survey)\b"
        _P6 = rf"\bresults?\b.*\b{_PART}\b|\b{_DOC}\b.*\b{_PART}\b"

        _study_like_re = re.compile(
            rf"(?:{_P1}|{_P2}|{_P3}|{_P4}|{_P5}|{_P6})", re.IGNORECASE)

        kept = []
        for q in questions:
            if not isinstance(q, str) or not q.strip():
                continue
            qn = _normalize(q)

            m = _aux_re.match(qn)
            if not m:
                continue

            # Trim any junk before the auxiliary
            qn = qn[m.start():]

            # Structural checks
            if not qn.endswith("?"):
                continue
            if len(qn.split()) < 3:
                continue
            if qn.lower().startswith(_followup_openers):
                continue

            # Study/doc/sample-like filters
            if _study_like_re.search(qn):
                continue

            kept.append(qn)

        return kept

    def _evaluate_pair(self, question, a_s, source_chunk, target_chunk, topic, subquery, path_save=None, save=True):
        is_relevant, _ = self._check_is_relevant(question, target_chunk)

        if is_relevant == 0:
            a_t = self.cannot_answer_dft
        else:
            a_t, _ = self._generate_answer(question, target_chunk)

        if "cannot answer the question" in a_t.lower():
            discrepancy_label = "NOT_ENOUGH_INFO"
            reason = self.cannot_answer_dft
        elif "cannot answer" in a_t.lower() or "personal opinion" in a_t.lower():
            discrepancy_label = "NOT_ENOUGH_INFO"
            reason = self.cannot_answer_personal
        else:
            discrepancy_label, reason = self._check_contradiction(
                question, a_s, a_t)

        # if discrepancy_label in ["CONTRADICTION", "CULTURAL_DISCREPANCY", "NOT_ENOUGH_INFO", "NO_DISCREPANCY"]:
        self._log_contradiction(
            topic,
            source_chunk,
            target_chunk,
            question,
            a_s,
            a_t,
            discrepancy_label,
            reason
        )

        if save and path_save is not None:
            self._print_result(discrepancy_label, question, a_s,
                               a_t, reason, target_chunk.text, source_chunk.text)

            question_id = len(self.questions_id[topic])
            self.questions_id[topic].add(question_id)

            self.results.append({
                "topic": topic,
                "question_id": question_id,
                "question": question,
                "subquery": subquery,
                "source_chunk": source_chunk.text,
                "target_chunk": target_chunk.text,
                "a_s": a_s,
                "a_t": a_t,
                "label": discrepancy_label,
                "reason": reason,
                # add original metadata
                "source_chunk_id": getattr(source_chunk, "id", None),
                "target_chunk_id": getattr(target_chunk, "id", None),
            })
            # save results every 200 entries
            if len(self.results) % 200 == 0:

                checkpoint = len(self.results) // 200
                results_checkpoint_path = Path(
                    f"{path_save}/results_topic_{topic}_{checkpoint}.parquet")
                discarded_checkpoint_path = Path(
                    f"{path_save}/discarded_topic_{topic}_{checkpoint}.parquet")

                df = pd.DataFrame(self.results)
                df_discarded = pd.DataFrame(self.discarded)

                df.to_parquet(results_checkpoint_path, index=False)
                df_discarded.to_parquet(discarded_checkpoint_path, index=False)

                # delete previous checkpoints
                old_results_checkpoint_path = Path(
                    f"{path_save}/results_topic_{topic}_{checkpoint-1}.parquet")
                old_discarded_checkpoint_path = Path(
                    f"{path_save}/discarded_topic_{topic}_{checkpoint-1}.parquet")

                if old_results_checkpoint_path.exists():
                    old_results_checkpoint_path.unlink()
                if old_discarded_checkpoint_path.exists():
                    old_discarded_checkpoint_path.unlink()

        return a_t, discrepancy_label, reason

    def _print_result(self, label, question, a_s, a_t, reason, target_text, source_text):
        color_map = {
            "CONTRADICTION": Fore.RED,
            "CULTURAL_DISCREPANCY": Fore.MAGENTA,
            "NOT_ENOUGH_INFO": Fore.YELLOW,
            "AGREEMENT": Fore.GREEN,
        }
        color = color_map.get(label, Fore.CYAN)

        print()
        print(
            f"{color}{Style.BRIGHT}== DISCREPANCY DETECTED: {label} =={Style.RESET_ALL}")
        print(f"{Fore.BLUE}Source Chunk Text:{Style.RESET_ALL} {source_text}")
        print(f"{Fore.BLUE}Question:{Style.RESET_ALL} {question}")
        print(f"{Fore.GREEN}Original Answer:{Style.RESET_ALL} {a_s}")
        print(f"{Fore.RED}Target Answer:{Style.RESET_ALL} {a_t}")
        print(f"{Fore.CYAN}Reason:{Style.RESET_ALL} {reason}")
        print(f"{Fore.YELLOW}Target Chunk Text:{Style.RESET_ALL} {target_text}")
        print()

    def _generate_questions(self, chunk):
        template_formatted = self.prompts["question_generation"].format(
            passage=chunk.text,
            full_document=extend_to_full_sentence(
                chunk.full_doc, 100) + " [...]",
        )

        response, _ = self._prompter.prompt(
            question=template_formatted,
            dry_run=self.dry_run
        )
        if self.dry_run:
            return [response], ""

        if "N/A" in response:
            return [], response

        try:
            for sep in ["\n", ","]:
                if sep in response:
                    questions = [
                        q.strip() for q in response.split(sep)
                        if q.strip() and "passage" not in q
                    ]
                    # remove
                    len_q = len(questions)
                    questions = self._filter_bad_questions(questions)
                    self._logger.info(
                        f"Filtered out {len_q - len(questions)} / {len_q} NO yes/no questions.")

                    return questions, ""
            return [], "No valid separator found"

        except Exception as e:
            self._logger.error(f"Error parsing questions: {e}")
            return [], str(e)

    def _generate_subqueries(self, question, chunk):
        try:
            template_formatted = self.prompts["subquery_generation"].format(
                question=question,
                passage=chunk.text,
            )
        except KeyError as e:
            self._logger.error(f"Missing field in subquery template: {e}")
            return [], str(e)

        response, _ = self._prompter.prompt(
            question=template_formatted,
            dry_run=self.dry_run
        )

        if self.dry_run:
            return [response], ""

        try:
            queries = [el.strip() for el in response.split(";") if el.strip()]
            return queries, ""
        except Exception as e:
            self._logger.error(f"Error extracting subqueries: {e}")
            return [], str(e)

    def retrieve_relevant_chunks(self, subquery: str, chunk):
        if self.dry_run:
            return []

        return self.target_corpus.retrieve_relevant_chunks(
            query=subquery,
            theta_query=chunk.top_k,
        )

    def _generate_answer(self, question, chunk):
        template_formatted = self.prompts["answer_generation"].format(
            question=question,
            passage=chunk.text,
            full_document=(extend_to_full_sentence(
                chunk.full_doc, 100) + " [...]")
        )

        response, _ = self._prompter_answer.prompt(
            question=template_formatted,
            dry_run=self.dry_run
        )

        if self.dry_run:
            return response, ""

        return response, ""

    def _check_is_relevant(self, question, chunk):
        template_formatted = self.prompts["relevance_checking"].format(
            passage=chunk.text,
            question=question
        )

        response, _ = self._prompter.prompt(
            question=template_formatted,
            dry_run=self.dry_run
        )

        if self.dry_run:
            return response, ""

        relevance = 1 if "yes" in response.lower() else 0

        return relevance, response

    def _check_contradiction(self, question, answer_s, answer_t):
        template_formatted = self.prompts["contradiction_checking"].format(
            question=question,
            answer_1=answer_s,
            answer_2=answer_t
        )

        response, _ = self._prompter.prompt(
            question=template_formatted,
            dry_run=self.dry_run
        )

        if self.dry_run:
            return response, ""

        label, reason = None, None
        lines = response.splitlines()
        for line in lines:
            if line.startswith("DISCREPANCY_TYPE:"):
                label = line.split("DISCREPANCY_TYPE:")[1].strip()
            elif line.startswith("REASON:"):
                reason = line.split("REASON:")[1].strip()

        # import pdb; pdb.set_trace()
        if label is None or reason is None:
            try:
                discrepancy_split = response.split("\n")
                reason = discrepancy_split[0].strip(
                    "\n").strip("REASON:").strip()
                label = discrepancy_split[1].strip(
                    "\n").strip("DISCREPANCY_TYPE:").strip()
            except:
                label = response
                reason = ""

        return self._clean_contradiction(label), reason

    def _clean_contradiction(self, discrepancy_label):
        corrections = {
            "NO_ DISCREPANCY": "NO_DISCREPANCY",
            "CULTURAL_ DISCREPANCY": "CULTURAL_DISCREPANCY"
        }
        for wrong, right in corrections.items():
            discrepancy_label = discrepancy_label.replace(wrong, right)
        return discrepancy_label

    def _check_entailement(self, textA, textB, threshold=0.5):
        """
        Compute textual entailment between (textA -> textB) using a 2-class MNLI head.

        Returns:
            entail_prob (float), contradict_prob (float), entails_bool (bool)
        """
        if self._nli_tokenizer is None or self._nli_model is None:
            self._logger.error("NLI tokenizer/model not available.")
            return 0.0, 0.0, False

        try:
            inputs = self._nli_tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=[(textA, textB)],
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                # neutral already removed (2 classes)
                logits = self._nli_model(**inputs).logits
                # [P(entail), P(contradict)]
                probs = torch.softmax(logits, dim=-1)[0]
                entail_prob = float(probs[0].item())
                contradict_prob = float(probs[1].item())
                entails = entail_prob >= threshold
            return entail_prob, contradict_prob, entails
        except Exception as e:
            self._logger.error(f"Entailment check failed: {e}")
            return 0.0, 0.0, False

    def _log_contradiction(self, topic, source_chunk, target_chunk, question, source_answer, target_answer, discrepancy_label, reason):
        self._logger.info({
            "topic": topic,
            "source_chunk_id": getattr(source_chunk, "id", None),
            "target_chunk_id": getattr(target_chunk, "id", None),
            "question": question,
            "original_answer": source_answer,
            "target_answer": target_answer,
            "discrepancy_label": discrepancy_label,
            "reason": reason
        })
