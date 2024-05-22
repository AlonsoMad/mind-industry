import os
import pandas as pd
import pyarrow.parquet as pq

def combine_parquet_files(input_dir, output_file):
    combined_df = pd.DataFrame()
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".parquet"):
            file_path = os.path.join(input_dir, filename)
            df = pd.read_parquet(file_path)
            if not df.empty:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    if not combined_df.empty:
        combined_df.to_parquet(output_file)
        print(combined_df.head())
        print(combined_df.columns)
        print(len(combined_df))
        print(f"Combined parquet file saved as: {output_file}")
    else:
        print("No non-empty parquet files found to combine.")

if __name__ == "__main__":
    input_directory = "/fs/nexus-scratch/lcalvo/rosie/data/es/translated"
    output_parquet_file = "/fs/nexus-scratch/lcalvo/rosie/data/corpus_pass_es_tr.parquet"
    
    combine_parquet_files(input_directory, output_parquet_file)
