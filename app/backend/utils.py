import os
import shutil
import pandas as pd

from flask import jsonify


def validate_and_get_dataset(email: str, dataset: str, output: str, phase: str):
    try:
        parquet_path = "/data/datasets_stage_preprocess.parquet"
        if not os.path.exists(parquet_path):
            print('No paths found')
            return jsonify({
                "status": "error",
                "message": f"Parquet file not found at {parquet_path}"
            }), 404

        df = pd.read_parquet(parquet_path)

        row = df[
            (df["Usermail"] == email) &
            (df["Dataset"] == dataset) &
            (df["Stage"] == 1)
        ]

        if row.empty:
            print('No dataset found')
            return jsonify({
                "status": "error",
                "message": f"No dataset found for user '{email}', dataset '{dataset}', stage 1."
            }), 404

        if phase == '1_Segmenter':
            dataset_path = f"/data/{email}/1_Preprocess/{dataset}/dataset"
            output_dir = f"/data/{email}/1_Preprocess/{dataset}/{phase}/{output}"
        
        elif phase == '2_Translator':
            dataset_path = f"/data/{email}/1_Preprocess/{dataset}/1_Segmenter/{output}/dataset"
            output_dir = f"/data/{email}/1_Preprocess/{dataset}/{phase}/{output}"
        
        elif phase == '3_Preparer':
            dataset_path = f"/data/{email}/1_Preprocess/{dataset}/2_Translator/{output}"
            output_dir = f"/data/{email}/2_TopicModelling/{output}"
        
        else:
            print('No phase found')
            return jsonify({
                "status": "error",
                "message": f"No dataset found for user '{email}', dataset '{dataset}', stage 1."
            }), 404
        
        if os.path.exists(f"{output_dir}/dataset"):
            return jsonify({
                "status": "error",
                "message": f"Output already exists at {output_dir}. Please choose another output name."
            }), 400
        os.makedirs(output_dir)

        return dataset_path, output_dir

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Validation error: {e}"
        }), 500
    
def validate_and_get_datasetTM(email: str, dataset: str, output: str):
    try:
        parquet_path = "/data/datasets_stage_preprocess.parquet"
        if not os.path.exists(parquet_path):
            print('No paths found')
            return jsonify({
                "status": "error",
                "message": f"Parquet file not found at {parquet_path}"
            }), 404

        df = pd.read_parquet(parquet_path)

        row = df[
            (df["Usermail"] == email) &
            (df["Dataset"] == dataset) &
            (df["Stage"] == 2)
        ]

        if row.empty:
            print('No dataset found')
            return jsonify({
                "status": "error",
                "message": f"No dataset found for user '{email}', dataset '{dataset}', stage 1."
            }), 404

        dataset_path = f"/data/{email}/2_TopicModelling/{dataset}/dataset"
        output_dir = f"/data/{email}/2_TopicModelling/{dataset}/{output}/"
        
        if os.path.exists(f"{output_dir}"):
            return jsonify({
                "status": "error",
                "message": f"Output already exists at {output_dir}. Please choose another output name."
            }), 400
        os.makedirs(output_dir)

        return dataset_path, output_dir

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Validation error: {e}"
        }), 500
    
def cleanup_output_dir(email: str, dataset: str, output: str):
    phases = ["1_Segmenter", "2_Translator", "3_Preparer"]
    
    for phase in phases:
        try:
            if phase == "3_Preparer": output_dir = f"/data/{email}/2_TopicModelling/{output}/"
            else: output_dir = f"/data/{email}/1_Preprocess/{dataset}/{phase}/"
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                print(f"[CLEANUP] Removed incomplete dataset at: {output_dir}")
        except Exception as e:
            print(f"[CLEANUP ERROR] Could not remove {output_dir}: {e}")

def aggregate_row(email: str, dataset: str, stage: int, output: str):
    try:
        parquet_path = "/data/datasets_stage_preprocess.parquet"
        if not os.path.exists(parquet_path):
            print('No paths found')
            return jsonify({
                "status": "error",
                "message": f"Parquet file not found at {parquet_path}"
            }), 404

        df = pd.read_parquet(parquet_path)

        new_row_data = {
            "Usermail": email,
            "Dataset": dataset,
            "Stage": stage,
            "Path": output
        }
        new_row_df = pd.DataFrame([new_row_data])
        df = pd.concat([df, new_row_df], ignore_index=True)

        df.to_parquet(parquet_path)

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({
                "status": "error",
                "message": str(e)
            }), 404
    
def get_download_dataset(email: str, dataset: str):
    try:
        parquet_path = "/data/datasets_stage_preprocess.parquet"
        if not os.path.exists(parquet_path):
            print('No paths found')
            return jsonify({
                "status": "error",
                "message": f"Parquet file not found at {parquet_path}"
            }), 404

        df = pd.read_parquet(parquet_path)

        row = df[
            (df["Usermail"] == email) &
            (df["Dataset"] == dataset) &
            (df["Stage"] == 3)
        ]

        if row.empty:
            print('No dataset found')
            return jsonify({
                "status": "error",
                "message": f"No dataset found for user '{email}', dataset '{dataset}', stage 1."
            }), 404

        dataset_path = f"/data/{email}/3_Download/{dataset}/dataset"
        return dataset_path

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Validation error: {e}"
        }), 500
    