import logging
from pathlib import Path
import pandas as pd # type: ignore
import numpy as np
from scipy import sparse
from src.mind.retriever import IndexRetriever
from src.utils.utils import init_logger


class Chunk:
    """A class representing a chunk of text with an ID, text content, and optional metadata. Optional metadata incudes (**so far**) the top k topics for the chunk as a list of tuples (topic_id, theta_weight), and the score of the chunk in the retrieval system (if used as a result of a retrieval query).
    """
    def __init__(self, id, text, full_doc=None, metadata=None):
        self.id = id
        self.text = text
        self.full_doc = full_doc
        self.metadata = metadata

    def __repr__(self):
        return f"Chunk(id={self.id}, text='{self.text[:30]}...')"


class Corpus:
    def __init__(
        self,
        df: pd.DataFrame,
        id_col="chunk_id",
        passage_col="chunk_text",
        full_doc_col="full_doc",
        config_path: Path = None,
        logger: logging.Logger = None,
        retriever: IndexRetriever = None
    ):  
        """ 
        Initializes a Corpus object from a pandas DataFrame. The DataFrame should contain the following columns:
        - id_col: The ID of the chunk (default: "chunk_id")
        - passage_col: The text of the chunk (default: "chunk_text")
        - full_doc_col: The full document text (default: "full_doc")
        The columns are renamed to "doc_id", "text", and "full_doc" respectively, for generalization to later use.
        
        If a retriever is provided, it will be used to retrieve relevant chunks from the corpus, that is, if a retriever is given, the corpus is a target corpus.
        
        Parameters
        ----------
        df: pd.DataFrame
            The DataFrame containing the corpus data.
        id_col: str
            The name of the column containing the chunk IDs (default: "chunk_id").
        passage_col: str
            The name of the column containing the chunk text (default: "chunk_text").
        full_doc_col: str
            The name of the column containing the full document text (default: "full_doc").
        config_path: Path
            The path to the configuration file (default: None).
        logger: logging.Logger
            The logger to use for logging (default: None).
        retriever: IndexRetriever
            The retriever to use for retrieving relevant chunks (default: None).
        """
        for col in [id_col, passage_col, full_doc_col]:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in dataframe")
        self.df = df.copy()
        # rename columns to "doc_id", "text", and "full_doc"
        self.df.rename(columns={passage_col: "text", full_doc_col: "full_doc"}, inplace=True)
        
        self._logger = logger if logger else init_logger(config_path, __name__)
        self._logger.info(f"Corpus initialized with {len(df)} documents.")
        self.retriever = retriever

    @classmethod
    def from_parquet_and_thetas(
        cls,
        path_parquet,
        path_thetas,
        id_col="chunk_id",
        passage_col="chunk_text",
        full_doc_col="full_doc",
        config_path=None,
        logger=None,
        language_filter="EN",
        retriever=None
    ):
        logger = logger if logger else init_logger(config_path, __name__)
        logger.info(f"Loading documents from {path_parquet}")
        logger.info(f"Loading topic distribution from {path_thetas}")

        df = pd.read_parquet(path_parquet)
        if language_filter:
            df = df[df[id_col].str.contains(language_filter)].copy()

        thetas = sparse.load_npz(path_thetas).toarray()
        df["thetas"] = list(thetas)
        df["top_k"] = df["thetas"].apply(lambda x: cls.get_doc_top_tpcs(x, topn=10))
        df["main_topic_thetas"] = df["thetas"].apply(lambda x: int(np.argmax(x)))

        logger.info(f"Loaded {len(df)} documents after filtering.")
        return cls(df, config_path=config_path, logger=logger, retriever=retriever, id_col=id_col, passage_col=passage_col, full_doc_col=full_doc_col)

    @staticmethod
    def get_doc_top_tpcs(doc_distr, topn=10):
        sorted_tpc_indices = np.argsort(doc_distr)[::-1]
        top = sorted_tpc_indices[:topn].tolist()
        return [(k, float(doc_distr[k])) for k in top if doc_distr[k] > 0]

    def chunks_with_topic(self, topic_id, sample_size=None):
        df_topic = self.df[self.df.main_topic_thetas == topic_id]
        if sample_size:
            self._logger.info(f"Sampling {sample_size} chunks for topic {topic_id}")
            df_topic = df_topic.sample(n=sample_size, random_state=42).reset_index(drop=True)

        self._logger.info(f"Found {len(df_topic)} chunks for topic {topic_id}")
        for _, row in df_topic.iterrows():
            yield Chunk(
                id=row["doc_id"],
                text=row["text"],
                full_doc=row.get("full_doc", ""),
                metadata={"top_k": row["top_k"]}
            )

    def retrieve_relevant_chunks(self, query: str, theta_query=None):
        if self.retriever is None:
            raise RuntimeError("No retriever has been set for this corpus.")

        results, _ = self.retriever.retrieve(
            query=query,
            theta_query=theta_query
        )
        chunks = []
        for result in results:
            try:
                row = self.df[self.df.doc_id == result["doc_id"]].iloc[0]
                chunk = Chunk(
                    id=result["doc_id"],
                    text=row["text"],
                    full_doc=row.get("full_doc", ""),
                    metadata={"score": result["score"], "top_k": row["top_k"]}
                )
                chunks.append(chunk)
            except KeyError:
                self._logger.warning(f"doc_id {result['doc_id']} not found in dataframe")

        return chunks