"""
MIND Profiling - Benchmark Data Generator

Generates synthetic benchmark data for profiling tests.
Creates realistic corpus data with controllable sizes.
"""

import random
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np


class BenchmarkDataGenerator:
    """Generates synthetic benchmark data for MIND profiling."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample vocabulary for generating realistic text
        self.topics = [
            "health", "technology", "environment", "economy", "education",
            "politics", "science", "culture", "sports", "travel"
        ]
        
        self.sentence_templates = [
            "The {topic} sector has seen significant changes in recent years.",
            "Research indicates that {topic} plays a crucial role in modern society.",
            "Experts believe that {topic} will continue to evolve rapidly.",
            "Many countries are investing heavily in {topic} initiatives.",
            "The impact of {topic} on daily life cannot be underestimated.",
            "Studies show that {topic} affects millions of people worldwide.",
            "The relationship between {topic} and quality of life is complex.",
            "Innovation in {topic} has led to numerous breakthroughs.",
            "Public opinion on {topic} varies significantly across regions.",
            "The future of {topic} depends on several key factors.",
        ]
        
        self.adjectives = [
            "significant", "important", "major", "critical", "essential",
            "growing", "emerging", "challenging", "complex", "dynamic"
        ]
        
        self.nouns = [
            "research", "development", "investment", "policy", "strategy",
            "analysis", "framework", "approach", "methodology", "system"
        ]
    
    def _generate_paragraph(self, topic: str, num_sentences: int = 5) -> str:
        """Generate a realistic paragraph about a topic."""
        sentences = []
        for _ in range(num_sentences):
            template = random.choice(self.sentence_templates)
            sentence = template.format(topic=topic)
            
            # Add some variation
            if random.random() > 0.5:
                adj = random.choice(self.adjectives)
                noun = random.choice(self.nouns)
                sentence += f" This {adj} {noun} requires careful consideration."
            
            sentences.append(sentence)
        
        return " ".join(sentences)
    
    def _generate_document(self, num_paragraphs: int = 3) -> str:
        """Generate a multi-paragraph document."""
        topic = random.choice(self.topics)
        paragraphs = [
            self._generate_paragraph(topic, random.randint(3, 7))
            for _ in range(num_paragraphs)
        ]
        return "\n\n".join(paragraphs)
    
    def generate_corpus(
        self,
        num_documents: int,
        output_path: Optional[Path] = None,
        with_translations: bool = False,
        with_thetas: bool = False,
        num_topics: int = 10,
    ) -> pd.DataFrame:
        """
        Generate a synthetic corpus for benchmarking.
        
        Parameters:
            num_documents: Number of documents to generate
            output_path: Path to save the corpus (parquet)
            with_translations: Include translated text columns
            with_thetas: Include topic distribution data
            num_topics: Number of topics for theta generation
        
        Returns:
            DataFrame with the generated corpus
        """
        print(f"Generating {num_documents} synthetic documents...")
        
        data = []
        langs = ["EN", "ES"] if with_translations else ["EN"]
        
        for i in range(num_documents):
            lang = random.choice(langs)
            doc_text = self._generate_document()
            
            row = {
                "id_preproc": f"{lang}_{i}",
                "chunk_id": f"{lang}_{i}",
                "doc_id": f"doc_{i // 10}",
                "text": doc_text,
                "full_doc": doc_text,
                "lang": lang,
            }
            
            # Add lemmas (simplified - just lowercase and split)
            row["lemmas"] = " ".join(doc_text.lower().split())
            
            if with_translations:
                row["lemmas_tr"] = row["lemmas"]  # Simplified
            
            if with_thetas:
                # Generate random topic distribution
                theta = np.random.dirichlet(np.ones(num_topics))
                row["top_k"] = [
                    (int(k), float(theta[k]))
                    for k in np.argsort(theta)[::-1][:5]
                ]
                row["main_topic_thetas"] = int(np.argmax(theta))
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if output_path:
            output_path = Path(output_path)
            df.to_parquet(output_path, compression="gzip", index=False)
            print(f"Saved corpus to: {output_path}")
            print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return df
    
    def generate_topic_model_outputs(
        self,
        num_documents: int,
        num_topics: int = 10,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """
        Generate synthetic topic model outputs for benchmarking.
        
        Creates:
            - thetas.npz: Document-topic distribution matrix
            - betas.npy: Topic-word distribution matrix
            - vocab.txt: Vocabulary file
        """
        from scipy import sparse
        
        output_dir = output_dir or self.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating topic model outputs for {num_documents} documents, {num_topics} topics...")
        
        # Generate thetas (document-topic distribution)
        thetas = np.random.dirichlet(np.ones(num_topics) * 0.5, size=num_documents)
        thetas_sparse = sparse.csr_matrix(thetas)
        
        thetas_path = output_dir / "thetas.npz"
        sparse.save_npz(thetas_path, thetas_sparse)
        
        # Generate betas (topic-word distribution)
        vocab_size = 5000
        betas = np.random.dirichlet(np.ones(vocab_size) * 0.01, size=num_topics)
        betas_path = output_dir / "betas.npy"
        np.save(betas_path, betas)
        
        # Generate vocabulary
        vocab_path = output_dir / "vocab.txt"
        with open(vocab_path, "w") as f:
            for i in range(vocab_size):
                freq = random.randint(1, 1000)
                f.write(f"word_{i}\t{freq}\n")
        
        print(f"Generated topic model outputs in: {output_dir}")
        
        return {
            "thetas_path": thetas_path,
            "betas_path": betas_path,
            "vocab_path": vocab_path,
        }
    
    def generate_faiss_index_data(
        self,
        num_vectors: int,
        dimension: int = 384,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Generate synthetic embeddings for FAISS benchmarking.
        """
        print(f"Generating {num_vectors} embeddings of dimension {dimension}...")
        
        embeddings = np.random.randn(num_vectors, dimension).astype(np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        if output_path:
            output_path = Path(output_path)
            np.save(output_path, embeddings)
            print(f"Saved embeddings to: {output_path}")
        
        return embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic benchmark data for MIND profiling"
    )
    parser.add_argument("--output-dir", type=str, default="aux_scripts/profiling/benchmark_data",
                       help="Output directory for generated data")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000],
                       help="Corpus sizes to generate")
    parser.add_argument("--with-thetas", action="store_true",
                       help="Include topic distribution data")
    parser.add_argument("--with-translations", action="store_true",
                       help="Include translation columns")
    
    args = parser.parse_args()
    
    generator = BenchmarkDataGenerator(Path(args.output_dir))
    
    for size in args.sizes:
        output_path = Path(args.output_dir) / f"corpus_{size}.parquet"
        generator.generate_corpus(
            num_documents=size,
            output_path=output_path,
            with_thetas=args.with_thetas,
            with_translations=args.with_translations,
        )
