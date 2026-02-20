# MIND: Multilingual Inconsistent Notion Detection

MIND is a powerful, user-in-the-loop AI pipeline designed to systematically detect contradictions and discrepancies within text and contextual databases. 

As AI agents and large context databases become central to enterprise operations, a fundamental question arises: *"How can my agents trust my data if it is not consistent?"* MIND addresses this by pruning and maintaining absolute contextual hygiene, ensuring that knowledge bases are free of contradictions and serve as reliable backbones for agentic workflows.

<p align="center">
  <img src="figures_tables/Raupi5.png" alt="MIND pipeline" width="100%">
</p>

- [MIND: Multilingual Inconsistent Notion Detection](#mind-multilingual-inconsistent-notion-detection)
  - [About MIND](#about-mind)
  - [Documentation](#documentation)
  - [**Installation**](#installation)
    - [Steps for deployment with uv](#steps-for-deployment-with-uv)
    - [Steps for deployment with docker](#steps-for-deployment-with-docker)
  - [Run MIND pipeline](#run-mind-pipeline)
  - [ROSIE-MIND](#rosie-mind)
  - [Replication of ablation experiments](#replication-of-ablation-experiments)
    - [Question and Answering](#question-and-answering)
    - [Retrieval](#retrieval)
    - [Discrepancies](#discrepancies)
  - [**Use cases**](#use-cases)
    - [**Wikipedia**](#wikipedia)
  - [**Data**](#data)
  - [Contributing](#contributing)
  - [License](#license)

## About MIND

MIND (Multilingual Inconsistent Notion Detection) detects contradictions and factual discrepancies in corporate and technical databases. The system is designed to work with large context DBs to ensure contextual hygiene for AI agents.

**Key capabilities:**
- Multi-LLM support (OpenAI, Ollama, vLLM, llama.cpp)
- Hybrid retrieval combining topic-based and embedding-based methods
- Web application for interactive database analysis
- Polylingual Topic Modeling to filter the DB by themes.
- Contextual pruning to identify redundant or noisy data

**Applications** include large corporate environments, technical documentation maintenance, specialized knowledge base verification, and context DB cleansing for agentic workflows.

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Technical Documentation](docs/technical-documentation.md)** (1,129 lines): Complete technology stack, architecture, module implementation details, configuration management, and deployment instructions. Intended for developers and software architects.

- **[Functional Documentation](docs/functional-documentation.md)** (679 lines): Project purpose, research methodology, use cases, ablation studies, and academic contributions. Intended for researchers and stakeholders.

- **[Architecture Diagrams](docs/architecture-diagrams.md)** (30+ Mermaid diagrams): Visual representations of system architecture, data pipelines, module dependencies, and deployment structure.

---

## **Installation**

We recommend **uv** for installing the necessary dependencies.

### Option 1: Development Setup (with uv)

1. **Clone the repository** (include submodules):

    ```bash
    git clone --recurse-submodules https://github.com/lcalvobartolome/mind.git
    cd mind
    ```

    If you already cloned without `--recurse-submodules`, run:

    ```bash
    git submodule update --init --recursive
    ```

2. **Install uv** by following the [official guide](https://docs.astral.sh/uv/getting-started/installation/).

3. **Create a local environment** (uses Python version from `pyproject.toml`):

    ```bash
    uv venv .venv
    ```

4. **Activate the environment**:

    ```bash
    source .venv/bin/activate   # On Linux/macOS
    .venv\Scripts\activate      # On Windows
    ```

5. **Install dependencies**:

    ```bash
    uv pip install -e .
    ```

6. **Verify the installation**:

    ```bash
    python -c "import mind; print(mind.__version__)"
    ```

---

### Option 2: Web Application Deployment (with Docker)

The MIND web application provides an intuitive interface for data preprocessing, topic modeling, and discrepancy analysis.

**Access Options**:
- **Hosted**: [https://mind.uc3m.es](https://mind.uc3m.es)
- **Local Deployment**: Follow steps below

1. **Clone the repository** (include submodules):

    ```bash
    git clone --recurse-submodules https://github.com/lcalvobartolome/mind.git
    cd mind
    ```

    If you already cloned without `--recurse-submodules`, run:

    ```bash
    git submodule update --init --recursive
    ```

2. **Build the containers**:

    ```bash
    docker compose build
    ```

3. **Start the services** in detached mode:

    ```bash
    docker compose up -d
    ```

4. **Access the application**:

    - **Frontend**: http://localhost:5050
    - **Auth Service**: http://localhost:5002
    - **Backend**: http://localhost:5001
    - **Database (PostgreSQL)**: port 5444 (mapped to 5432 in container)

For detailed deployment instructions, see [`app/README.md`](app/README.md) or the [Technical Documentation](docs/technical-documentation.md).

## Run MIND pipeline

To run the MIND pipeline, you need a collection of **loosely aligned** documents (e.g., corresponding Wikipedia articles in different languages). These do not need to be perfect translations—just share similar topics.

### Pipeline Steps

#### 1. Preprocess Corpora

The `mind.corpus_building` module provides scripts for segmenting documents, creating loose alignments (translation), NLP preprocessing, and assembling the final dataset for the PLTM wrapper.

```bash
# Segment documents into passages
python3 src/mind/corpus_building/segmenter.py \
  --input INPUT_PATH \
  --output OUTPUT_PATH \
  --text_col TEXT_COLUMN \
  --id_col ID_COLUMN

# Translate passages from anchor to comparison language (and vice versa)
python3 src/mind/corpus_building/translator.py \
  --input INPUT_PATH \
  --output OUTPUT_PATH \
  --src_lang SRC_LANG \
  --tgt_lang TGT_LANG \
  --text_col TEXT_COLUMN \
  --lang_col LANG_COLUMN

# Preprocess and prepare the final DataFrame for the pipeline
python3 src/mind/corpus_building/data_preparer.py \
  --anchor ANCHOR_PATH \
  --comparison COMPARISON_PATH \
  --output OUTPUT_PATH \
  --schema SCHEMA_JSON_OR_PATH
```

**Programmatic Usage**: See the [Wikipedia use case](use_cases/wikipedia/generate_dtset.py) for a complete workflow example.

---

#### 2. Train a PLTM Model

Train a Polylingual Topic Model on the prepared dataset:

```bash
python3 src/mind/topic_modeling/polylingual_tm.py \
  --input PREPARED_DATASET_PATH \
  --lang1 LANG1 \
  --lang2 LANG2 \
  --model_folder MODEL_OUTPUT_DIR \
  --num_topics NUM_TOPICS \
  [+ additional optional params]
```

Run `python3 src/mind/topic_modeling/polylingual_tm.py --help` for all options.

---

#### 3. (Optional) Label Topics

Assign meaningful labels to discovered topics:

```bash
python3 src/mind/topic_modeling/topic_label.py \
  --lang1 LANG1 \
  --lang2 LANG2 \
  --model_folder MODEL_OUTPUT_DIR \
  [+ additional optional params]
```

Run `python3 src/mind/topic_modeling/topic_label.py --help` for all options.

---

#### 4. Run the MIND Pipeline

Detect discrepancies and perform downstream analysis:

```bash
python3 src/mind/pipeline/cli.py \
    --src_corpus_path SRC_CORPUS_PATH \
    --src_thetas_path SRC_THETAS_PATH \
    --src_id_col SRC_ID_COL \
    --src_passage_col SRC_PASSAGE_COL \
    --src_full_doc_col SRC_FULL_DOC_COL \
    --src_lang_filter SRC_LANG \
    --tgt_corpus_path TGT_CORPUS_PATH \
    --tgt_thetas_path TGT_THETAS_PATH \
    --tgt_id_col TGT_ID_COL \
    --tgt_passage_col TGT_PASSAGE_COL \
    --tgt_full_doc_col TGT_FULL_DOC_COL \
    --tgt_lang_filter TGT_LANG \
    --topics TOPIC_IDS \
    --path_save RESULTS_DIR \
    [+ additional optional params]
```

- `--topics` should be a comma-separated list of topic IDs, e.g., `--topics 15,17`
- Run `python3 src/mind/pipeline/cli.py --help` for all options

**For detailed pipeline workflow**, see [Architecture Diagrams - MIND Pipeline](docs/architecture-diagrams.md#5-mind-pipeline-workflow).

---

## ROSIE-MIND

**ROSIE-MIND** is a dataset created by subsampling topics 12 (*Pregnancy*), 15 (*Infant Care*), and 25 (*Pediatric Healthcare*), and annotating them in two batches:

- **ROSIE-MIND-v1**: Generated using the *quora-distilbert-multilingual* embedding model and *qwen:32b* LLM. Contains 80 annotated samples.
- **ROSIE-MIND-v2**: Generated using *BAAI/bge-m3* embeddings and *llama3.3:70b* LLM. Contains 651 annotated samples.

The complete ROSIE-MIND dataset is available on [Hugging Face](https://huggingface.co/datasets/lcalvobartolome/rosie_mind).

## Replication of ablation experiments

### Question and Answering

1. **Generate answers** for questions using different LLMs:

    ```bash
    ./bash_scripts/run_answering_disc.sh
    ```

2. **Prepare human evaluation task**:

    ```bash
    python3 ablation/qa/prepare_eval_task.py
    ```

    Evaluation dimensions:
    - **Questions**: Verifiability, Passage Independence, Clarity, Terminology, Self-Containment, Naturalness
    - **Answers**: Faithfulness, Passage Dependence, Passage Reference Avoidance, Structured Response, Language Consistency

3. **Generate tables and figures**:

    See [ablation/qa/get_figures.ipynb](ablation/qa/get_figures.ipynb) for analysis and visualizations.

---

### Retrieval

1. **Run retrieval** to get relevant passages:

    ```bash
    ./bash_scripts/run_retrieval.sh
    ```

2. **Get gold passages** for metric calculation:

    ```bash
    python3 ablation/retrieval/get_gold_passages.py
    ```

    A passage is considered relevant only if all four LLMs agree.

3. **Run statistical tests and generate tables.**

    This step performs statistical analysis and generates summary tables for the retrieval experiments.

    ```bash
    ./bash_scripts/generate_tables.sh
    ```

### Discrepancies

1. **Run discrepancy detection on the controlled dataset (FEVER-DPLACE-Q).**

    Benchmarks the discrepancy detection module against a fixed reference set.

    ```bash
    python3 ablation/discrepancies/run_disc_ablation_controlled.py
    ```

2. **Prepare evaluation data from MIND-detected discrepancies and FEVER-DPLACE-Q.**

    Builds a standardized annotation file for human evaluation and downstream analysis.

    ```bash
    python3 ablation/discrepancies/prepare_eval_task.py
    ```

3. **Generate tables and figures.**

    Analyze the results and create publication-ready tables and figures for the discrepancy evaluation. This [notebook](ablation/discrepancies/get_figures_tables.ipynb) summarizes the main findings and visualizations used in the paper.

## **Use cases**

### **Wikipedia**

To evaluate the **generalizability** of MIND — both **across languages** and **beyond the health domain** — we applied the pipeline to a collection of **German–English Wikipedia article pairs**.  

The `use_cases.wikipedia` module provides scripts for:

- **Retrieving** parallel or loosely aligned Wikipedia pages,  
- **Preprocessing** the corpora, and  
- **Training** topic models to run the full MIND pipeline.

#### 1. Retrieve bilingual Wikipedia data

Retrieve a dataset of German–English Wikipedia articles, either **aligned** (same topic in both languages) or **partially aligned** (based on a specified alignment percentage) and preprocess the retrieved data and prepare it for topic modeling and pipeline execution:

```bash
python3 -m use_cases.wikipedia.generate_dtset --output-path test
```

This script runs all necessary **retrieving**, **cleaning**, **segmentation**, and **structuring** steps, producing a dataset ready for model training.

#### 2. Train models and run the MIND pipeline

Once the dataset is prepared, you can train various models and apply the full **MIND pipeline** using the provided bash script:

```bash
./bash_scripts/run_en_de.sh
```

This script demonstrates an **end-to-end workflow**, from dataset preparation to discrepancy detection, for the German–English Wikipedia use case.

For more details on the preprocessing and pipeline stages, see the [Run MIND pipeline](#run-mind-pipeline) section above.

##  **Data**

All data used and generated for the paper, including ROSIE-MIND and the synthetic dataset FEVER-DPLACE-Q (with LLM annotations), are available at the [HuggingFace collection](https://huggingface.co/collections/lcalvobartolome/mind-data-68e2a690025b4dc28c5e8458). For both ROSIE and ENDE datasets (see use cases section), we include the preprocessed corpora augmented with topic modeling information ready to apply MIND. Topic words in each language and topic labels are available as additional metadata. Data from ablation studies is available as well as complete output from MIND pipeline executions on ROSIE and ENDE is available upon request.

## Contributing

Contributions are welcome. For bug reports and feature requests, please use [GitHub Issues](https://github.com/lcalvobartolome/mind/issues). For code contributions, submit a pull request.

If you use MIND in your research, please cite:

```bibtex
@article{calvo2024mind,
  title={Discrepancy Detection at the Data Level: Toward Consistent Multilingual Question Answering},
  author={Calvo-Bartolomé, Lorena and Madroñal, Alonso},
  year={2024}
}
```

## License

This project is licensed under the MIT License. Copyright (c) 2024 Lorena Calvo-Bartolomé. See [LICENSE](LICENSE) for details.

---

**Additional Resources:**
- Main Repository: [GitHub](https://github.com/lcalvobartolome/mind)
- Datasets: [HuggingFace Collection](https://huggingface.co/collections/lcalvobartolome/mind-data-68e2a690025b4dc28c5e8458)
- Web Application: [https://mind.uc3m.es](https://mind.uc3m.es)
- Comprehensive Documentation: [`docs/`](docs/) directory