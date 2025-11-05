import os
import json
import uuid

from pathlib import Path
from utils import cleanup_output_dir, aggregate_row
from flask import Blueprint, request, jsonify, current_app, send_file, after_this_request


preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix='/preprocessing')

TASKS = {}


def run_step(step_name, func, app, *args, **kwargs):
    step_id = str(uuid.uuid4())
    
    def target():
        with app.app_context():
            TASKS[step_id] = {'status': 'running', 'message': f"{step_name} in progress", 'name': step_name}
            try:
                func(*args, **kwargs)
                TASKS[step_id]['status'] = 'completed'
                TASKS[step_id]['message'] = f"{step_name} completed successfully!"
            except Exception as e:
                TASKS[step_id]['status'] = 'error'
                TASKS[step_id]['message'] = str(e)
    
    from threading import Thread
    Thread(target=target).start()
    
    TASKS[step_id] = {'status': 'pending', 'message': f"{step_name} task created", 'name': step_name}
    return step_id

@preprocessing_bp.route('/status/<step_id>', methods=['GET'])
def status(step_id):
    task = TASKS.get(step_id)
    if not task:
        return jsonify({"status": "not_found", "message": "Step ID not found"}), 404
    return jsonify(task), 200


@preprocessing_bp.route('/segmenter', methods=['POST'])
def segmenter():
    try:
        data = request.get_json()
        email = data.get("email")
        dataset = data.get("dataset")
        segmenter_data = data.get("segmenter_data")

        def do_segment():
            from utils import validate_and_get_dataset
            from mind.corpus_building.segmenter import Segmenter
            
            validation = validate_and_get_dataset(
                email=email,
                dataset=dataset,
                output=segmenter_data['output'],
                phase="1_Segmenter"
            )
            if isinstance(validation, tuple):
                dataset_path, output_dir = validation
            else:
                raise Exception("Validation failed")

            print(f'Segmenting dataset {output_dir}...')

            try:
                seg = Segmenter(config_path="/src/config/config.yaml")
                seg.segment(
                    path_df=dataset_path,
                    path_save=f'{output_dir}/dataset',
                    text_col=segmenter_data['text_col'],
                    min_length=segmenter_data['min_length'],
                    sep=segmenter_data['sep']
                )

                print(f'Finalize segmenting dataset {output_dir}')

            except Exception as e:
                print(str(e))
                cleanup_output_dir(email, dataset, segmenter_data['output'])
                raise e

        step_id = run_step("Segmenting", do_segment, app=current_app._get_current_object())
        return jsonify({"step_id": step_id, "message": "Segmenter task started"}), 200

    except Exception as e:
        print(str(e))
        cleanup_output_dir(email, dataset, segmenter_data['output'])
        return jsonify({"status": "error", "message": str(e)}), 500

@preprocessing_bp.route('/translator', methods=['POST'])
def translator():
    try:
        data = request.get_json()
        email = data.get("email")
        dataset = data.get("dataset")
        translator_data = data.get("translator_data")

        def do_translate():
            from utils import validate_and_get_dataset
            from mind.corpus_building.translator import Translator
            
            validation = validate_and_get_dataset(
                email=email,
                dataset=dataset,
                output=translator_data['output'],
                phase="2_Translator"
            )
            if isinstance(validation, tuple):
                dataset_path, output_dir = validation
            else:
                raise Exception("Validation failed")
            
            # To operate correctly
            translator_data['text_col'] = "full_doc"

            # Translator
            print(f"Translating dataset {dataset}...")

            try:
                os.system(f'cp /data/{email}/1_Preprocess/{dataset}/1_Segmenter/{translator_data["output"]}/dataset {output_dir}/dataset_{translator_data["tgt_lang"]}2{translator_data["src_lang"]}')
                os.system(f'cp /data/{email}/1_Preprocess/{dataset}/dataset {output_dir}/dataset_{translator_data["src_lang"]}2{translator_data["tgt_lang"]}')
                
                trans = Translator(config_path="/src/config/config.yaml")
                
                # First src -> tgt
                # trans.translate(
                #     path_df=dataset_path,
                #     save_path=f'{output_dir}/dataset_{translator_data["tgt_lang"]}2{translator_data["src_lang"]}',
                #     src_lang=translator_data['src_lang'],
                #     tgt_lang=translator_data['tgt_lang'],
                #     text_col=translator_data['text_col'],
                #     lang_col=translator_data['lang_col'],
                # )

                # # Second tgt -> src
                # trans.translate(
                #     path_df=dataset_path,
                #     save_path=f'{output_dir}/dataset_{translator_data["src_lang"]}2{translator_data["tgt_lang"]}',
                #     src_lang=translator_data['tgt_lang'],
                #     tgt_lang=translator_data['src_lang'],
                #     text_col=translator_data['text_col'],
                #     lang_col=translator_data['lang_col'],
                # )

                print(f'Finalize translating dataset {output_dir}')
            
            except Exception as e:
                print(str(e))
                cleanup_output_dir(email, dataset, translator_data['output'])
                raise e

        step_id = run_step("Translating", do_translate, app=current_app._get_current_object())
        return jsonify({"step_id": step_id, "message": "Translator task started"}), 200

    except Exception as e:
        print(str(e))
        cleanup_output_dir(email, dataset, translator_data['output'])
        return jsonify({"status": "error", "message": str(e)}), 500


@preprocessing_bp.route('/preparer', methods=['POST'])
def preparer():
    try:
        data = request.get_json()
        email = data.get("email")
        dataset = data.get("dataset")
        preparer_data = data.get("preparer_data")

        def do_preparer():
            from utils import validate_and_get_dataset
            from mind.corpus_building.data_preparer import DataPreparer
            
            validation = validate_and_get_dataset(
                email=email,
                dataset=dataset,
                output=preparer_data['output'],
                phase="3_Preparer"
            )
            if isinstance(validation, tuple):
                dataset_path, output_dir = validation
            else:
                raise Exception("Validation failed")

            # Data Preparer
            print(f"Preparing dataset: {dataset}")

            try:
                nlpipe_json = {
                        "id": f"{preparer_data["schema"]["chunk_id"]}",
                        "raw_text": f"{preparer_data["schema"]["text"]}",
                        "title": ""
                    }

                with open("/backend/NLPipe/config.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                data['mind'] = nlpipe_json

                with open("/backend/NLPipe/config.json", 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                prep = DataPreparer(
                    preproc_script="/backend/NLPipe/src/nlpipe/cli.py",
                    config_path="/backend/NLPipe/config.json",
                    config_logger_path="/src/config/config.yaml",
                    stw_path="/backend/NLPipe/src/nlpipe/stw_lists",
                    spacy_models={
                        "en": "en_core_web_sm",
                        "de": "de_core_news_sm",
                        "es": "es_core_news_sm"},
                    schema=preparer_data['schema'],
                )

                prep.format_dataframes(
                    anchor_path=f'{dataset_path}/dataset_{preparer_data["src_lang"]}2{preparer_data["tgt_lang"]}',
                    comparison_path=f'{dataset_path}/dataset_{preparer_data["tgt_lang"]}2{preparer_data["src_lang"]}',
                    path_save=f'{output_dir}/dataset'
                )

                os.remove(f"temp_{preparer_data['output']}.json")
                os.rmdir(f"{output_dir}/_tmp_preproc")

                aggregate_row(email, preparer_data['output'], 2, f'{output_dir}/dataset')

                print(f'Finalize preparing dataset {output_dir}')
            
            except Exception as e:
                print(str(e))
                cleanup_output_dir(email, dataset, preparer_data['output'])
                os.remove(f"temp_{preparer_data['output']}.json")
                os.rmdir(f"{output_dir}/_tmp_preproc")
                raise e

        step_id = run_step("Data Preparer", do_preparer, app=current_app._get_current_object())
        return jsonify({"step_id": step_id, "message": "Data Preparer task started"}), 200

    except Exception as e:
        print(str(e))
        cleanup_output_dir(email, dataset, preparer_data['output'])
        os.remove(f"temp_{preparer_data['output']}.json")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@preprocessing_bp.route('/topicmodeling', methods=['POST'])
def topicmodelling():
    try:
        data = request.get_json()
        email = data.get("email")
        dataset = data.get("dataset")
        output = data.get('output')
        lang1 = data.get('lang1')
        lang2 = data.get('lang2')
        k = data.get('k')

        def train_topicmodel():
            from utils import validate_and_get_datasetTM
            from mind.topic_modeling.polylingual_tm import PolylingualTM
            
            validation = validate_and_get_datasetTM(
                email=email,
                dataset=dataset,
                output=output
            )
            if isinstance(validation, tuple):
                dataset_path, output_dir = validation
            else:
                raise Exception("Validation failed")

            print(f'Training model (k = {k}) for dataset {dataset_path}...')

            try:
                model = PolylingualTM(
                    lang1=lang1,
                    lang2=lang1,
                    model_folder=Path(output_dir),
                    num_topics=int(k),
                    mallet_path="/backend/Mallet/bin/mallet",
                    add_stops_path="/src/mind/topic_modeling/stops"
                )

                model.train(dataset_path)

                aggregate_row(email, output, 3, output_dir)

                print('Finalize train model')

            except Exception as e:
                print(str(e))
                raise e

        step_id = run_step("TopicModeling", train_topicmodel, app=current_app._get_current_object())
        return jsonify({"step_id": step_id, "message": "Training Topic Model task started"}), 200

    except Exception as e:
        print(str(e))
        return jsonify({"status": "error", "message": str(e)}), 500
    

@preprocessing_bp.route("/download", methods=["POST"])
def download_data():
    data = request.get_json()
    stage = data.get("stage")
    email = data.get("email")
    dataset = data.get("dataset")

    if not dataset:
        return jsonify({"message": "Data missing"}), 400

    try:
        from utils import get_download_dataset
        dataset_path = get_download_dataset(int(stage), email, dataset)

        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({"message": "Failed to generate data"}), 500
        
        if dataset_path.endswith(".zip"):
            @after_this_request
            def remove_file(response):
                try:
                    os.remove(dataset_path)
                except Exception as e:
                    preprocessing_bp.logger.error(f"Error removing temporary file: {e}")
                return response

        return send_file(
            dataset_path,
            as_attachment=True,
            download_name=os.path.basename(dataset_path),
            mimetype='application/octet-stream'
        )

    except Exception as e:
        return jsonify({"message": f"Error generating data for download: {str(e)}"}), 500
