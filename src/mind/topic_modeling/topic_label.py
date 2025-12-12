from pathlib import Path
from dotenv import dotenv_values
from collections import defaultdict
from mind.prompter.prompter import Prompter
from mind.utils.utils import init_logger, load_prompt, load_yaml_config_file


class TopicLabel(object):
    def __init__(
        self,
        lang1: str,
        lang2: str,
        model_folder: str,
        llm_model: str,
        llm_server: str = None,
        config_path: Path = Path("config/config.yaml"),
        logger=None,
        env_path=None,
    ) -> None:
        """Initialize the TopicLabel object.

        Parameters
        ----------
        lang1 : str
            The first language in the corpus.
        lang2 : str
            The second language in the corpus.
        model_folder : str
            The folder where all information related to the model will be stored.

        """

        self._logger = logger if logger else init_logger(config_path, __name__)

        self.config = load_yaml_config_file(config_path, "mind", self._logger)
        
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

        path = self.config.get("prompts", {}).get("topic_label")
        if path is None:
            raise ValueError("Missing prompt path for: topic_label")
        self._prompt_label = load_prompt(path)

        self._lang1 = lang1
        self._lang2 = lang2
        self._model_folder = model_folder

        if not self._model_folder.exists():
            self._logger.error(
                f"-- -- Given model folder {self._model_folder} does not exist. Please create a Topic Model first."
            )
            raise ValueError(f"Given model folder {self._model_folder} does not exist.")

    def _load_corpus(self, corpus_file: str) -> dict:
        """
        Load a text corpus from a file into a dictionary mapping document IDs to text.

        Parameters
        ----------
        corpus_file : str
            Path to a text file where each line represents a document with the format:
            `<doc_id> <optional_value> <raw_text>`.

        Returns
        -------
        corpus_dict : dict
            Dictionary with document IDs as keys (int) and raw text as values (str).
        """
        
        corpus_dict = {}
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=2)
                doc_id = int(float(parts[0]))
                try:
                    corpus_dict[doc_id] = parts[2]
                except:
                    corpus_dict[doc_id] = ''
        return corpus_dict
    
    def _most_representative_docs(self, path_topic: str) -> dict:
        """
        Retrieve the top 5 most representative documents per topic for a bilingual corpus.

        Parameters
        ----------
        path_topic : str
            Path to the folder containing the Mallet topic output files, including `doc-topics.txt`.

        Returns
        -------
        final_dict : dict
            Nested dictionary with structure `{lang: {topic_id: concatenated_texts}}`.
            Each topic contains the concatenated texts of its top 5 documents in the respective language.
        """
        
        topic_docs = defaultdict(list)

        # doc proportions
        with open(f'{path_topic}/doc-topics.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                doc_id = int(parts[0])
                topic_props = parts[1:]
                for i in range(0, len(topic_props), 2):
                    topic = int(topic_props[i])
                    prop = float(topic_props[i+1])
                    topic_docs[topic].append((doc_id, prop))

        # top 5 docs per topic
        top_docs_per_topic_id = {}
        for topic, doc_list in topic_docs.items():
            doc_list_sorted = sorted(doc_list, key=lambda x: x[1], reverse=True)
            top_docs_per_topic_id[topic] = doc_list_sorted[:5]

        corpus_lang1 = self._load_corpus(f'{self._model_folder}/train_data/corpus_{self._lang1}.txt')
        corpus_lang2 = self._load_corpus(f'{self._model_folder}/train_data/corpus_{self._lang2}.txt')

        final_dict = {self._lang1: {}, self._lang2: {}}

        # top 5 doc text per lang
        for topic, doc_list in top_docs_per_topic_id.items():
            texts_lang1 = [corpus_lang1[doc_id] for doc_id, _ in doc_list if doc_id in corpus_lang1]
            texts_lang2 = [corpus_lang2[doc_id] for doc_id, _ in doc_list if doc_id in corpus_lang2]
            final_dict[self._lang1][topic] = '\n'.join(texts_lang1)
            final_dict[self._lang2][topic] = '\n'.join(texts_lang2)
        
        return final_dict
    
    def label_topic(self) -> None:
        """
        Generate human-readable labels for topics in a bilingual Mallet topic model using a prompt-based approach.

        Reads the top keywords per topic and the most representative documents, then uses a prompt template
        to generate topic labels for both languages. Labels are saved in `labels_<lang>.txt` files.
        """
        
        path_topic = f'{self._model_folder}/mallet_output/'
        topic_labels = {
            self._lang1: [],
            self._lang2: []
        }

        # lang1
        topic_keys = defaultdict(list)
        with open(f'{path_topic}/keys_{self._lang1}.txt', 'r', encoding='utf-8') as f:
            topic_keys[self._lang1] = [line.strip() for line in f]

        # lang2
        with open(f'{path_topic}/keys_{self._lang2}.txt', 'r', encoding='utf-8') as f:
            topic_keys[self._lang2] = [line.strip() for line in f]

        top_docs_per_topic = self._most_representative_docs(path_topic)

        # Prompting lang1
        for k in range(len(topic_keys[self._lang1])):
            template_formatted = self._prompt_label.format(
                keywords=topic_keys[self._lang1][k],
                docs='\n'.join(top_docs_per_topic[self._lang1][k])
            )
            res, _ = self._prompter.prompt(template_formatted)
            print(res)
            topic_labels[self._lang1].append(res)

        # Prompting lang2
        for k in range(len(topic_keys[self._lang2])):
            template_formatted = self._prompt_label.format(
                keywords=topic_keys[self._lang2][k],
                docs='\n'.join(top_docs_per_topic[self._lang2][k])
            )
            res, _ = self._prompter.prompt(template_formatted)
            print(res)
            topic_labels[self._lang2].append(res)

        # Write labels in model_folder
        with open(f'{path_topic}/labels_{self._lang1}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(topic_labels[self._lang1]))
        
        with open(f'{path_topic}/labels_{self._lang2}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(topic_labels[self._lang2]))
