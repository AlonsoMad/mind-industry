import time
import uuid
import threading

from flask import Blueprint, request, jsonify, current_app


preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix='/preprocessing')

TASKS = {}

def run_in_background(func, step_id, app, *args, **kwargs):
    TASKS[step_id] = {'status': 'pending', 'message': f"{func.__name__} task created"}
    
    def target():
        with app.app_context():
            try:
                TASKS[step_id]['status'] = 'running'
                TASKS[step_id]['message'] = f"{func.__name__} in progress..."
                func(*args, **kwargs)
                TASKS[step_id]['status'] = 'completed'
                TASKS[step_id]['message'] = f"{func.__name__} completed successfully!"
            except Exception as e:
                TASKS[step_id]['status'] = 'error'
                TASKS[step_id]['message'] = str(e)
                print(e)
    
    threading.Thread(target=target).start()
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
            seg = Segmenter(config_path="/src/config/config.yaml")
            seg.segment(
                path_df=dataset_path,
                path_save=output_dir,
                text_col=segmenter_data['text_col'],
                min_length=segmenter_data['min_length'],
                sep=segmenter_data['sep']
            )

            print(f'Finalize segmenting dataset {output_dir}')

        step_id = run_in_background(do_segment, step_id=str(uuid.uuid4()), app=current_app._get_current_object())
        return jsonify({"step_id": step_id, "message": "Segmenter task started"}), 200

    except Exception as e:
        print(e)
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

            # Translator
            print(f"Translating dataset {dataset}...")

            # trans = Translator(config_path="/src/config/config.yaml")
            # trans.translate(
            #     path_df=dataset_path,
            #     save_path=output_dir,
            #     src_lang=translator_data['src_lang'],
            #     tgt_lang=translator_data['tgt_lang'],
            #     text_col=translator_data['text_col'],
            #     lang_col=translator_data['lang_col'],
            # )

            time.sleep(5)
            print(f'Finalize translating dataset {output_dir}')

        step_id = run_in_background(do_translate, step_id=str(uuid.uuid4()), app=current_app._get_current_object())
        return jsonify({"step_id": step_id, "message": "Segmenter task started"}), 200

    except Exception as e:
        print(e)
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

            # prep = DataPreparer(
            #     preproc_script="/backend/NLPipe/src/nlpipe/cli.py",
            #     config_path="/src/config/config.yaml",
            #     stw_path="/backend/NLPipe/src/nlpipe/stw_lists",
            #     spacy_models={
            #         "en": "en_core_web_sm",
            #         "de": "de_core_news_sm",
            #         "es": "es_core_news_sm"},
            #     schema={
            #         "chunk_id": "id_preproc",
            #         "doc_id": "id",
            #         "text": "text",
            #         "full_doc": "summary",
            #         "lang": "lang",
            #         "title": "title",
            #         "url": "url",
            #         "equivalence": "equivalence",
            #     },
            # )

            # prep.format_dataframes(
            #     anchor_path=str(output_dir / f"{args.base_name}_trans_s{args.seed_lang}_t{args.target_lang}_n{args.ndocs}.parquet.gzip"),
            #     comparison_path=str(output_dir / f"{args.base_name}_trans_s{args.target_lang}_t{args.seed_lang}_n{args.ndocs}.parquet.gzip"),
            #     path_save=str(output_dir / f"{args.base_name}_polylingual_s{args.seed_lang}_t{args.target_lang}_n{args.ndocs}.parquet.gzip")
            # )

            time.sleep(5)

            print(f'Finalize preparing dataset {output_dir}')

        step_id = run_in_background(do_preparer, step_id=str(uuid.uuid4()), app=current_app._get_current_object())
        return jsonify({"step_id": step_id, "message": "Segmenter task started"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
