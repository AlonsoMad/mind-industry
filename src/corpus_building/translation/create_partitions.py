import argparse
import pathlib
import pandas as pd

def main():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input_file",
        type=str,
        required=False,
        default="data/source/corpus_rosie/corpus_strict_v3.0_en_compiled_passages_lang.parquet"
    )
    argparser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        default="data/source/corpus_rosie/translated/en"
    )   
    argparser.add_argument(
        "--samples_per_split",
        type=int,
        required=False,
        default=15000
    )
    
    args = argparser.parse_args()
    
    # Read dataset
    input_file = pathlib.Path(args.input_file)
    df = pd.read_parquet(args.input_file)
    
    # Calculate the number of splits required
    num_splits = (len(df) + args.samples_per_split - 1) // args.samples_per_split

    # Split the DataFrame into chunks of samples_per_split rows each
    split_dfs = [df[i*args.samples_per_split:(i+1)*args.samples_per_split] for i in range(num_splits)]

    # Write each split DataFrame to a separate Parquet file
    output_folder = pathlib.Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for i, split_df in enumerate(split_dfs):
        path_save = output_folder.joinpath(f"{input_file.stem}_{i+1}.parquet")
        split_df.to_parquet(path_save)   
        
    print(f"-- -- Split {input_file} into {num_splits} files of {args.samples_per_split} samples each.")
    print(f"-- -- Saved to {output_folder}: ")
    [print(f"-- -- --> {path}") for path in output_folder.iterdir() if path.is_file() and path.suffix == ".parquet"]

if __name__ == "__main__":
    main()
