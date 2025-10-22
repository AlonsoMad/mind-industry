import os
import pandas as pd
from flask import jsonify


def validate_and_get_dataset(email: str, dataset: str, output: str, phase: str):
    try:
        parquet_path = "/data/datasets_stage_preprocess.parquet"
        if not os.path.exists(parquet_path):
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
            return jsonify({
                "status": "error",
                "message": f"No dataset found for user '{email}', dataset '{dataset}', stage 1."
            }), 404

        dataset_path = row.iloc[0]["Path"]

        if phase == '3_Preparer': output_dir = f"/data/{email}/2_TopicModelling/{output}/dataset"
        else: output_dir = f"/data/{email}/1_Preprocess/{dataset}/{phase}/{output}"
        
        if os.path.exists(f"{output_dir}/dataset"):
            return jsonify({
                "status": "error",
                "message": f"Output already exists at {output_dir}. Please choose another output name."
            }), 400
        os.makedirs(output_dir)

        return dataset_path, f"{output_dir}/dataset"

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Validation error: {e}"
        }), 500
    