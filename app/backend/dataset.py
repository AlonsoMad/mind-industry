import os
import shutil
import numpy as np
import pandas as pd

from flask import Blueprint, jsonify, request


datasets_bp = Blueprint("datasets", __name__)
DATASETS_STAGE = os.getenv("DATASETS_STAGE")


@datasets_bp.route("/create_user_folders", methods=["POST"])
def create_user_folders():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"error": "Missing email"}), 400

    base_path = f"/data/{email}"
    source_base = "/data/all"

    try:
        if not os.path.exists(source_base):
            return jsonify({"error": f"Source path {source_base} does not exist"}), 500

        os.makedirs(base_path, exist_ok=True)

        for folder_name in os.listdir(source_base):
            src_folder = os.path.join(source_base, folder_name)
            dst_folder = os.path.join(base_path, folder_name)
            if os.path.isdir(src_folder):
                shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)

        df = pd.read_parquet(f'{source_base}/datasets_stage_preprocess.parquet', engine='pyarrow')
        df['Usermail'] = email
        df['Path'] = df['Path'].str.replace("/all/", f"/{email}/", regex=False)
        
        df_data = pd.read_parquet('/data/datasets_stage_preprocess.parquet', engine='pyarrow')
        df_data = pd.concat([df_data, df], ignore_index=True)
        df_data.to_parquet('/data/datasets_stage_preprocess.parquet', engine='pyarrow', index=False)

    except Exception as e:
        print(str(e))
        return jsonify({"error": f"Failed to create user folders: {str(e)}"}), 500

    return jsonify({"message": "User folders created and parquet updated successfully"}), 200

@datasets_bp.route("/update_user_folders", methods=["POST"])
def update_user_folders():
    data = request.get_json()
    old_email = data.get("old_email")
    new_email = data.get("new_email")

    if not old_email or not new_email:
        return jsonify({"error": "Missing emails"}), 400

    old_path = f"/data/{old_email}"
    new_path = f"/data/{new_email}"
    folders = ["1_RawData", "2_PreprocessData", "3_TopicModel", "4_Detection"]

    try:
        os.makedirs(new_path, exist_ok=True)

        for folder in folders:
            old_folder = os.path.join(old_path, folder)
            new_folder = os.path.join(new_path, folder)
            if os.path.exists(old_folder):
                shutil.move(old_folder, new_folder)

        if os.path.exists(old_path) and not os.listdir(old_path):
            os.rmdir(old_path)

        if os.path.exists(DATASETS_STAGE):
            df = pd.read_parquet(DATASETS_STAGE, engine="pyarrow")
            print(df.keys())
            df["Usermail"] = df["Usermail"].replace(old_email, new_email)
            df.to_parquet(DATASETS_STAGE, index=False)

    except Exception as e:
        return jsonify({"error": f"Failed to update folders/parquet: {str(e)}"}), 500

    return jsonify({"message": "User folders and parquet updated successfully"}), 200

@datasets_bp.route("/datasets", methods=["GET"])
def get_datasets():
    email = request.args.get("email")
    folders = ["1_RawData", "2_PreprocessData", "3_TopicModel", "4_Detection"]
    path = f"/data/{email}"

    final_dataset_preview = []
    final_datasets_name = []
    final_shapes = []
    stages = []

    for i in range(len(folders)):
        dataset_path = f'{path}/{folders[i]}'
        if not os.path.exists(dataset_path):
            return jsonify({"error": f"Dataset path {dataset_path} does not exist."}), 404

        dataset_list, datasets_name, shapes = load_datasets(dataset_path, folders[i])

        # Convert DF into lists for JSON
        dataset_preview = [df.head(5).to_dict(orient="records") for df in dataset_list]

        final_dataset_preview.extend(dataset_preview)
        final_datasets_name.extend(datasets_name)
        final_shapes.extend(shapes)
        stages.extend([i + 1] * len(dataset_list))

    return jsonify({
        "datasets": final_dataset_preview,
        "names": final_datasets_name,
        "shapes": final_shapes,
        "stages": stages
    })

@datasets_bp.route("/datasets_detection", methods=["GET"])
def get_datasets_detection():
    email = request.args.get("email")

    try:
        df = pd.read_parquet(DATASETS_STAGE, engine='pyarrow')
    except Exception as e:
        return jsonify({"error": f"ERROR: {str(e)}"}), 500

    filtered_df = df[
        (df["Usermail"] == email) & (df["Stage"] == 3)
    ]

    dataset_detection = filtered_df.groupby('OriginalDataset')['Dataset'].apply(list).to_dict()

    return jsonify({
        "dataset_detection": dataset_detection
    })

@datasets_bp.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    file = request.files.get('file')

    data = request.args
    path = data.get('path')
    email = data.get('email')
    stage = data.get('stage')
    dataset_name = data.get('dataset_name')
    extension = data.get('extension')
    sep = data.get('sep')

    if not file or not path or not email or not stage or not dataset_name or not extension:
        return jsonify({'error': 'Missing file or arg'}), 400
    
    output_dir = f"/data/{path}/"

    new_data = {
        "Usermail": email,
        "Dataset": dataset_name,
        "OriginalDataset": None,
        "Stage": int(stage),
        "Path": f'{output_dir}dataset'
    }
    
    df = pd.read_parquet(DATASETS_STAGE)

    if ((df['Usermail'] == new_data['Usermail']) &
        (df['Dataset'] == new_data['Dataset']) & 
        (df['Stage'] == new_data['Stage'])
        ).any():
        print('Existing that dataset for that stage for that user')
        return jsonify({'error': 'Existing that dataset in that stage'}), 450


    print('Creating new dataset...')

    os.makedirs(output_dir, exist_ok=True)

    # Save file
    with open(f'{output_dir}/dataset', 'wb') as f:
        f.write(file.read())

    # In case dataset is CSV
    if extension == 'csv':
        try:
            if not sep:
                return jsonify({'error': 'Missing separator for csv file'}), 400
            
            df_csv = pd.read_csv(f'{output_dir}/dataset', sep=sep)
            df_csv.to_parquet(f'{output_dir}/dataset', engine='pyarrow')
        except Exception as e:
            print(e)
            shutil.rmtree(output_dir)
            return jsonify({'error': 'Couldn\'t save csv file'}), 400

    try:
        df_new_data = pd.DataFrame([new_data])
        df_final = pd.concat([df, df_new_data], ignore_index=True)
        df_final.to_parquet(DATASETS_STAGE)

        return jsonify({'message': "File processed and saved"})
    
    except Exception as e:
        print(e)
        shutil.rmtree(output_dir)
        return jsonify({'error': 'Couldn\'t save file'}), 400

def clean(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [clean(x) for x in obj]
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    return obj

def load_datasets(dataset_path: str, folder: str):
    dataset_list = []
    datasets_name = os.listdir(dataset_path)
    shapes = []

    for _, d in enumerate(datasets_name):
        try:
            ds_path = os.path.join(dataset_path, d, 'dataset')
            ds = pd.read_parquet(ds_path)
            shapes.append(list(ds.shape))
            if 'index' in ds.columns:
                ds = ds.drop(columns=['index'])
            dataset_list.append(ds)
        except Exception as e:
            dataset_list.append(pd.DataFrame())
            if folder == '4_Detection':
                k_list = []
                for k in os.listdir(os.path.join(dataset_path, d)):
                    k_list.append(k)
                shapes.append(k_list)
            else:
                shapes.append([0, 0])

    return dataset_list, datasets_name, shapes
