    # Run the pipeline
    for num_topics in [25]:#np.arange(5, 51, 5):
        #num_topics = 10
        model_folder = f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/models/1_{num_topics}"
        row_top_k = f"theta_{num_topics}_top_tpcs"
        
        source_corpus = {
            "corpus_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/final_fever_for_mind_7675.parquet",
            "id_col": "claim_evidence_id",
            "passage_col": "evidence", #"claim",
            "full_doc_col": "evidence", #"claim",
            "load_thetas": False,
            "row_top_k": row_top_k,
        }
        
        index_path = f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/index_corpus_train_chunked/{num_topics}k"
        Path(index_path).mkdir(parents=True, exist_ok=True)
        
        target_corpus = {
            "corpus_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/corpus_train_chunked.parquet",
            "thetas_path": f"{model_folder}/mallet_output/EN/thetas.npz",
            "id_col": "chunk_id",
            "passage_col": "chunk_text",
            "full_doc_col": "full_doc",
            "index_path": index_path,
            "load_thetas": True,
        }
        
        results_per_model = defaultdict(list)
        llm_models = ["qwen:32b"]#"qwen3:32b",
        for llm_model in llm_models:
            mind = MIND(
                llm_model=llm_model,#"qwen:32b",
                source_corpus=source_corpus,
                target_corpus=target_corpus,
                retrieval_method="TB-ENN",
                multilingual=False,
                lang="en",
                config_path=Path("config/config.yaml"),
                logger=None,
                dry_run=False
            )
            all_results = []
            topics = np.arange(0, num_topics)
            for topic in topics:
                # run pipeline for each topic
                print(f"Running pipeline for topic {topic}")
                mind.run_pipeline([topic])
                # save results as df
                results = pd.DataFrame(mind.results)
                
                # save intermediate results per topic
                results.to_parquet(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/results/mind_results_from_model_{num_topics}k_{llm_model}_upt_prompt_evidence_7675el_qwen3:32b_for_answer_topic_{topic}.parquet")

                #results.to_parquet(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/mind_results_{topic}_from_model_0.5{num_topics}k_{llm_model}.parquet")
                all_results.append(results)
            all_results = pd.concat(all_results)
            all_results.to_parquet(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/results/mind_results_from_model_{num_topics}k_{llm_model}_upt_prompt_evidence_7675el_qwen3:32b_for_answer.parquet")
            # save results for each model
            results_per_model[llm_model].append(results)