import pathlib
from src.topic_modeling.polylingual_tm import PolylingualTM


def main():
    # Train PolyLingual Topic Model
    model = PolylingualTM(
        lang1="EN",
        lang2="DE",
        model_folder= pathlib.Path("/Users/lbartolome/Documents/GitHub/LinQAForge/data/models/test1"),
        num_topics=8
    )
    
    df_path = pathlib.Path("/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/pltm/df.parquet")
    model.train(df_path)
    
if __name__ == "__main__":
    main()