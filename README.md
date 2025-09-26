# MIND

This repository contains the code and data for reproducing experiments from our paper Discrepancy Detection at the Data Level: Toward Consistent Multilingual Question Answering.

<p align="center">
  <img src="figures_tables/Raupi5.png" alt="MIND pipeline" width="100%">
</p>

- [MIND](#mind)
  - [**Installation**](#installation)
    - [Steps for deployment with uv](#steps-for-deployment-with-uv)
  - [Run MIND pipeline](#run-mind-pipeline)
  - [ROSIE-MIND](#rosie-mind)
  - [Replication of ablation experiments](#replication-of-ablation-experiments)
    - [Question and Answering](#question-and-answering)
    - [Retrieval](#retrieval)
    - [Discrepancies](#discrepancies)
  - [ Other data](#other-data)
    - [ROSIE](#rosie)
    - [ENDE corpus](#ende-corpus)
    - [FEVER-DPLACE-Q](#fever-dplace-q)
  - [Use cases](#use-cases)



## **Installation**

We recommend **uv** for installing the necessary dependencies.

### Steps for deployment with uv

1. Clone the repository (include submodules):

    ```bash
    git clone --recurse-submodules https://github.com/lcalvobartolome/mind.git
    cd mind
    ```

    If you already cloned without `--recurse-submodules`, run:

    ```bash
    git submodule update --init --recursive
    ```

2. Install uv by following the [official guide](https://docs.astral.sh/uv/getting-started/installation/).

3. Create a local environment (it will use the python version specified in pyproject.toml)

    ```bash
    uv venv .venv
    ```

4. Activate the environment:

    ```bash
    source .venv/bin/activate   # On Linux/macOS
    .venv\Scripts\activate      # On Windows
    ```

5. Install dependencies:

    ```bash
    uv pip install -e .
    ```

6. Verify the installation:

    ```bash
    python -c "import mind; print(mind.__version__)"
    ```

## Run MIND pipeline

To run the MIND pipeline, you need a collection of *loosely aligned* documents (e.g., corresponding Wikipedia articles in different languages). These do not need to be perfect translations—just share similar topics.

**Steps for running the pipeline:**

1. **Preprocess corpora:** The `mind.corpus_building` module provides scripts for segmenting documents, creating loose alignments (translation), NLP preprocessing, and assembling the final dataset for the PLTM wrapper.

    You can run these scripts directly from the command line with flexible arguments:

    ```bash
    # Segment documents into passages
    python3 src/mind/corpus_building/segmenter.py --input INPUT_PATH --output OUTPUT_PATH --text_col TEXT_COLUMN --lang_col LANG_COLUMN

    # Translate passages from anchor to comparison language (and vice versa)
    python3 src/mind/corpus_building/translator.py --input INPUT_PATH --output OUTPUT_PATH --src_lang SRC_LANG --tgt_lang TGT_LANG --text_col TEXT_COLUMN --lang_col LANG_COLUMN

    # Preprocess and prepare the final DataFrame for the pipeline
    python3 src/mind/corpus_building/data_preparer.py --anchor ANCHOR_PATH --comparison COMPARISON_PATH --output OUTPUT_PATH --schema SCHEMA_JSON_OR_PATH
    ```

    - Replace `INPUT_PATH`, `OUTPUT_PATH`, `TEXT_COLUMN`, `LANG_COLUMN`, `SRC_LANG`, `TGT_LANG`, `ANCHOR_PATH`, `COMPARISON_PATH`, and `SCHEMA_JSON_OR_PATH` with your actual file paths and column names.
    - The `--schema` argument for `data_preparer.py` can be a JSON string or a path to a JSON file mapping required columns.

    Alternatively, you can import and use these modules programmatically. See the [Wikipedia use case](use_cases/wikipedia/generate_dtset.py) for a complete example of how to use all these scripts in a workflow.
  
2. **Train a PLTM model:** Train a Polylingual Topic Model on the prepared dataset.

    ```bash
    python3 src/mind/topic_modeling/polylingual_tm.py \
      --input PREPARED_DATASET_PATH \
      --lang1 LANG1 \
      --lang2 LANG2 \
      --model_folder MODEL_OUTPUT_DIR \
      --num_topics NUM_TOPICS \
      [+ additional optional params]
    ```

    - Replace each argument (e.g., `PREPARED_DATASET_PATH`, `LANG1`, `LANG2`, `MODEL_OUTPUT_DIR`, `NUM_TOPICS`, etc.) with your actual file paths, language codes, and options.
    - See `python3 src/mind/topic_modeling/polylingual_tm.py --help` for full details and all available options.

3. **Run the MIND pipeline:** Detect discrepancies and perform downstream analysis.

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

    - Replace each argument (e.g., ``SRC_CORPUS_PATH``, ``TGT_CORPUS_PATH``, ``TOPIC_IDS``, etc.) with your actual file paths, column names, and options.
    - ``--topics`` should be a comma-separated list of topic IDs, e.g. ``--topics 15,17``.
    - See ``python3 src/mind/pipeline/cli.py --help`` for full details and all available options.


## ROSIE-MIND

**ROSIE-MIND** is a dataset created by subsampling topics 12 (*Pregnancy*), 15 (*Infant Care*), and 25 (*Pediatric Healthcare*), and annotating them in two batches:

- **ROSIE-MIND-v1**: Generated using the *quora-distilbert-multilingual* embedding model and *qwen:32b* LLM. Contains 80 annotated samples.
- **ROSIE-MIND-v2**: Generated using *BAAI/bge-m3* embeddings and *llama3.3:70b* LLM. Contains XX annotated samples (update with final count).

## Replication of ablation experiments

### Question and Answering

1. **Generate answers.**

    For each question generated by the MIND question generator (using different LLMs), generate answers and detect discrepancies. This step requires already generated questions and relevant passages (from the Retrieval step).åå

    ```bash
    ./bash_scripts/run_answering_disc.sh
    ```

2. **Prepare human evaluation task.**

    Prepare annotation files for human evaluation of both questions and answers. The evaluation is based on the following dimensions:

    - **Questions:** Verifiability, Passage Independence, Clarity, Terminology, Self-Containment, Naturalness
    - **Answers:** Faithfulness, Passage Dependence, Passage Reference Avoidance, Structured Response, Language Consistency

    ```bash
    python3 ablation/qa/prepare_eval_task.py
    ```

3. **Generate tables and figures.**

    Analyze the results and create publication-ready tables and figures for the discrepancy evaluation. This [notebook](ablation/qa/get_figures.ipynb) summarizes the main findings and visualizations used in the paper.

### Retrieval

1. **Run retrieval to get relevant passages for the questions.**

    For each question generated by the MIND question generator (using different LLMs), this step produces two Excel files per run: one with weighted and one with unweighted retrieval results.

    ```bash
    ./bash_scripts/run_retrieval.sh
    ```

2. **Get gold passages for retrieval metric calculation.**

    For each passage retrieved by any method (ANN, ENN, TB-ENN, TB-ANN), four LLMs independently rate its relevance to the question. A passage is considered relevant only if all four LLMs agree.

    ```bash
    python3 ablation/retrieval/get_gold_passages.py
    ```

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

##  Other data

### ROSIE

- **Corpus**
- **Model**
  
### ENDE corpus

- **Corpus**
- **Model**

### FEVER-DPLACE-Q


## Use cases

### Wikipedia

```bash
python3 -m wikipedia.generate_dtset --output-path test
```