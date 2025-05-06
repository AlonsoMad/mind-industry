from pathlib import Path
from colorama import Fore, Style
from typing import Union
from tqdm import tqdm # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from src.mind.corpus import Corpus
from src.mind.retriever import IndexRetriever
from src.mind.utils import extend_to_full_sentence
from src.prompter.prompter import Prompter
from src.utils.utils import init_logger, load_prompt, load_yaml_config_file

class MIND:
    """
    MIND pipeline class.
    
    =========================
    for k in topics:
        for c_s in source_corpus.chunks_with_topic(k):
            questions = generate_yes_no_questions(c_s)
            
            for p in questions:
                subqueries = generate_subqueries(p)
                
                for subq in subqueries:
                    c_t_chunks = retrieve_relevant_chunks(subq, target_corpus)
                    
                    for c_t in c_t_chunks:
                        a_s = generate_answer(c_o, subq)
                        a_t = generate_answer(c_t, subq)
                        contradiction = check_contradiction(a_o, a_t, p)

    =========================                
    All components are build based on the Prompter class. Each component has its own template and is responsible for generating the input for the next component.
    """
    def __init__(
        self,
        llm_model: str,
        source_corpus: Union[Corpus, dict],
        target_corpus: Union[Corpus, dict],
        retrieval_method: str = "TB-ENN",
        multilingual: bool = True,
        lang: str = "en",
        config_path: Path = Path("config/config.yaml"),
        logger=None,
        dry_run: bool = False,
    ):
        self._logger = logger if logger else init_logger(config_path, __name__)
        
        self.dry_run = dry_run
        self.retrieval_method = retrieval_method
        
        self.config = load_yaml_config_file(config_path, "mind", self._logger)
        self._embedding_model = self.config.get("embedding_models", {}).get("multilingual") if multilingual else self.config.get("embedding_models", {}).get("monolingual").get(lang, "sentence-transformers/all-MiniLM-L6-v2")
        self.cannot_answer_dft = self.config.get("cannot_answer_dft", "I cannot answer the question given the context.")
        self.cannot_answer_personal = self.config.get("cannot_answer_personal", "I cannot answer the question since the context only contains personal opinions.")

        self._prompter = Prompter(
            model_type=llm_model,
            config_path=config_path,
        )

        self.prompts = {}
        for name in ["question_generation", "subquery_generation", "answer_generation", "contradiction_checking", "relevance_checking"]:
            path = self.config.get("prompts", {}).get(name)
            if path is None:
                raise ValueError(f"Missing prompt path for: {name}")
            self.prompts[name] = load_prompt(path)

        self.source_corpus = self._init_corpus(source_corpus, is_target=False)
        self.target_corpus = self._init_corpus(target_corpus, is_target=True)

        if self.dry_run:
            self._logger.warning("Dry run mode is ON — no LLM calls will be made.")
            
        self.results = []

    def _init_corpus(
        self,
        corpus: Union[Corpus, dict],
        is_target: bool = False
    ) -> Corpus:
        
        if isinstance(corpus, Corpus):
            return corpus

        required_keys = {"corpus_path", "thetas_path", "id_col", "passage_col", "full_doc_col"}
        
        if not required_keys.issubset(corpus.keys()):
            raise ValueError(f"Missing keys in corpus config dict: {required_keys - corpus.keys()}")

        corpus_obj = Corpus.from_parquet_and_thetas(
            path_parquet=corpus["corpus_path"],
            path_thetas=Path(corpus["thetas_path"]),
            id_col=corpus["id_col"],
            passage_col=corpus["passage_col"],
            full_doc_col=corpus["full_doc_col"],
            language_filter=corpus.get("language_filter", None),
            logger=self._logger
        )

        if is_target:
            retriever = IndexRetriever(
                model=SentenceTransformer(self._embedding_model),
                logger=self._logger,
                top_k=self.config.get("mind", {}).get("top_k", 10),
                batch_size=self.config.get("mind", {}).get("batch_size", 32),
                min_clusters=self.config.get("mind", {}).get("min_clusters", 8),
                do_weighting=self.config.get("mind", {}).get("do_weighting", True),
            )
            retriever.build_or_load_index(
                source_path=corpus["corpus_path"],
                model_path=Path(corpus["thetas_path"]).parent.parent,
                save_path_parent=corpus["index_path"],
                method=corpus.get("method", "TB-ENN"),
                col_to_index=corpus["passage_col"],
                col_id=corpus["id_col"],
                lang=corpus.get("language_filter", None),
            )
            corpus_obj.retriever = retriever

        return corpus_obj
    
    def run_pipeline(self, topics, sample_size=None):
        for topic in topics:
            self._process_topic(topic, sample_size=sample_size)

    def _process_topic(self, topic, sample_size=None):
        for chunk in tqdm(self.source_corpus.chunks_with_topic(topic, sample_size), desc=f"Topic {topic}"):
            self._process_chunk(chunk, topic)

    def _process_chunk(self, chunk, topic):
        print(f"Processing chunk {chunk.id} for topic {topic}")
        questions, _ = self._generate_questions(chunk)
        if questions == []:
            print(f"{Fore.RED}No questions generated for chunk {chunk.id}{Style.RESET_ALL}")
            return
        print(f"Generated questions: {questions}\n")
        for question in questions:
            self._process_question(question, chunk, topic)

    def _process_question(self, question, source_chunk, topic):
        subqueries, _ = self._generate_subqueries(question, source_chunk)
        print(f"Generated subqueries: {subqueries}\n")
        a_s, _ = self._generate_answer(question, source_chunk)
        print(f"Generated original answer: {a_s}\n")

        for subquery in subqueries:
            target_chunks = self.target_corpus.retrieve_relevant_chunks(subquery, source_chunk.metadata["top_k"])
            for target_chunk in target_chunks:
                self._evaluate_pair(question, a_s, source_chunk, target_chunk, topic)

    def _evaluate_pair(self, question, a_s, source_chunk, target_chunk, topic):
        a_t, _ = self._generate_answer(question, target_chunk)
        is_relevant, _ = self._check_is_relevant(question, target_chunk)

        if is_relevant == 0:
            a_t = self.cannot_answer_dft

        if "cannot answer the question" in a_t.lower():
            discrepancy_label = "NOT_ENOUGH_INFO"
            reason = self.cannot_answer_dft
        elif "cannot answer" in a_t.lower() and "personal opinion" in a_t.lower():
            discrepancy_label = "NOT_ENOUGH_INFO"
            reason = self.cannot_answer_personal
        else:
            discrepancy_label, reason = self._check_contradiction(question, a_s, a_t)

        if discrepancy_label in ["CONTRADICTION", "CULTURAL_DISCREPANCY", "NOT_ENOUGH_INFO"]:
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
            self._print_result(discrepancy_label, question, a_s, a_t, reason)
            self.results.append({
                "topic": topic,
                "question": question,
                "source_chunk": source_chunk.text,
                "target_chunk": target_chunk.text,
                "a_s": a_s,
                "a_t": a_t,
                "label": discrepancy_label,
                "reason": reason,
            })

    def _print_result(self, label, question, a_s, a_t, reason):
        color_map = {
            "CONTRADICTION": Fore.RED,
            "CULTURAL_DISCREPANCY": Fore.MAGENTA,
            "NOT_ENOUGH_INFO": Fore.YELLOW,
            "AGREEMENT": Fore.GREEN,
        }
        color = color_map.get(label, Fore.CYAN)

        print()
        print(f"{color}{Style.BRIGHT}== DISCREPANCY DETECTED: {label} =={Style.RESET_ALL}")
        print(f"{Fore.BLUE}Question:{Style.RESET_ALL} {question}")
        print(f"{Fore.GREEN}Original Answer:{Style.RESET_ALL} {a_s}")
        print(f"{Fore.RED}Target Answer:{Style.RESET_ALL} {a_t}")
        print(f"{Fore.CYAN}Reason:{Style.RESET_ALL} {reason}")
        print()


    def _generate_questions(self, chunk):
        template_formatted = self.prompts["question_generation"].format(
            passage=chunk.text,
            full_document=extend_to_full_sentence(chunk.full_doc, 100) + " [...]",
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
            full_document=(extend_to_full_sentence(chunk.full_doc, 100)+ " [...]")
        )

        response, _ = self._prompter.prompt(
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
        

        if label is None or reason is None:
            try:
                discrepancy_split = response.split("\n")
                reason = discrepancy_split[0].strip("\n").strip("REASON:").strip()
                label = discrepancy_split[1].strip("\n").strip("DISCREPANCY_TYPE:").strip()
            except:
                label = response
                reason = ""
                
        return label, reason
    

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

if __name__ == "__main__":
    
    # Example usage
    mind = MIND(
        llm_model="qwen:32b",
        source_corpus={
            "corpus_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet", "thetas_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/26_jan_no_dup/poly_rosie_1_30/mallet_output/thetas_EN.npz",
            "id_col": "doc_id",
            "passage_col": "text",
            "full_doc_col": "full_doc",
            "language_filter": "EN",
            },
        target_corpus={
            "corpus_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet", "thetas_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/26_jan_no_dup/poly_rosie_1_30/mallet_output/thetas_ES.npz",
            "id_col": "doc_id",
            "passage_col": "text",
            "full_doc_col": "full_doc",
            "language_filter": "ES",
            "index_path": "test",
        },
        dry_run=False
    )
    
    # Run the pipeline
    topic = 0
    mind.run_pipeline([topic], sample_size=2)
    print(mind.results)