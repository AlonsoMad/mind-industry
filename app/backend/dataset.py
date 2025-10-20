from flask import Blueprint, jsonify, request
import os, glob
import pandas as pd
import numpy as np

datasets_bp = Blueprint("datasets", __name__)

@datasets_bp.route("/datasets", methods=["GET"])
def get_datasets():
    email = request.args.get("email")

    dataset_path = f"/data/{email}/1_Preprocess/"

    if not os.path.exists(dataset_path):
        return jsonify({"error": f"Dataset path {dataset_path} does not exist."}), 404

    dataset_list, datasets_name, shapes = load_datasets(dataset_path)

    # Convertimos cada dataframe a lista de diccionarios para JSON
    dataset_preview = [df.head(20).to_dict(orient="records") for df in dataset_list]

    return jsonify({
        "datasets": dataset_preview,
        "names": datasets_name,
        "shapes": shapes.tolist()
    })

@datasets_bp.route("/final_results/<og_dataset>", methods=["GET"])
def get_final_results(og_dataset):
    """
    Devuelve todos los resultados finales de un dataset original,
    incluyendo raw_text del dataset original.
    """
    dataset_path = os.getenv("DATASET_PATH", "/data/3_joined_data")
    og_dataset_path = os.path.join(dataset_path, og_dataset, 'polylingual_df')

    if not os.path.exists(og_dataset_path):
        return jsonify({"error": f"Original dataset {og_dataset} not found."}), 404

    try:
        og_ds = pd.read_parquet(og_dataset_path)
        if 'index' in og_ds.columns:
            og_ds = og_ds.drop(columns=['index'])
    except Exception as e:
        return jsonify({"error": f"Failed to load original dataset: {e}"}), 500

    final_results_path = os.path.join(dataset_path, "final_results")
    mind_info = []

    for file in glob.glob(os.path.join(final_results_path, "**", "*.*"), recursive=True):
        if file.endswith((".parquet", ".csv", ".xlsx")):
            try:
                if file.endswith(".parquet"):
                    df = pd.read_parquet(file)
                elif file.endswith(".csv"):
                    df = pd.read_csv(file)
                elif file.endswith(".xlsx"):
                    df = pd.read_excel(file)

                # Merge raw_text from original dataset
                if "doc_id" in df.columns and "doc_id" in og_ds.columns:
                    df = df.merge(
                        og_ds[["doc_id", "raw_text"]],
                        on="doc_id",
                        how="left"
                    )

                dataset_name = os.path.relpath(file, final_results_path)
                mind_info.append({
                    "dataset_name": dataset_name,
                    "data": df.to_dict(orient="records")
                })

            except Exception as e:
                print(f"⚠️ Failed to load {file}: {e}")

    return jsonify({"results": mind_info})


def load_datasets(dataset_path: str):
    dataset_list = []
    datasets_name = os.listdir(dataset_path)
    shapes = np.empty((len(datasets_name), 2), dtype=int)

    for i, d in enumerate(datasets_name):
        ds_path = os.path.join(dataset_path, d, 'dataset')
        try:
            ds = pd.read_parquet(ds_path)
            shapes[i] = ds.shape
            if 'index' in ds.columns:
                ds = ds.drop(columns=['index'])
            dataset_list.append(ds)
        except Exception as e:
            print(f"Error loading dataset {d}: {e}")
            dataset_list.append(pd.DataFrame())
            shapes[i] = (0, 0)

    return dataset_list, datasets_name, shapes
