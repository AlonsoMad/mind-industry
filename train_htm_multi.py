import argparse
import pathlib
from termcolor import colored
import time
from src.topic_modeling.hierarchical.hierarchical_tm import HierarchicalTM
from src.topic_modeling.polylingual_tm import PolylingualTM


def main(father_model, langs=["EN", "ES"]):
    # Define colors for each language
    lang_colors = {
        "EN": "red",
        "ES": "blue",
        # Add more languages and their colors as needed
    }

    # father model
    father_model = pathlib.Path(father_model)

    hmg = HierarchicalTM()

    # Load topic-keys in each lang
    all_keys = {}
    for lang in langs:
        # Default to white if color is not defined
        color = lang_colors.get(lang, "white")
        print(colored("#" * 50, color))
        print(colored(f"-- -- Topic keys in {lang.upper()}: ", color))
        print(colored("-" * 50, color))
        keys = []
        with (father_model / f"mallet_output/keys_{lang}.txt").open('r', encoding='utf8') as fin:
            keys = [el.strip() for el in fin.readlines()]
        all_keys[lang] = keys
        for id, tpc in enumerate(keys):
            print(colored(f"- Topic {id}: {tpc}", color))
        print("\n")

    # ask input from user: he needs to select a topic id
    topic_id = input(f"Please select the topic you want to expand: ")
    try:
        topic_id = int(topic_id)
        if topic_id < 0 or topic_id >= len(keys):
            raise ValueError("Topic id out of range.")
        print(f"Selected Topic {topic_id}: {all_keys[langs[0]][topic_id]}")
        for lang in langs:
            color = lang_colors.get(lang, "white")
            print(
                colored(f"Keys in {lang}: {all_keys[lang][topic_id]}", color))
    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    # htm version
    htm_version = input(f"Please select the method you want to use (htm_ws/htm_ds): ")
    if htm_version not in ["htm_ds", "htm_ws"]:
        raise ValueError("Invalid method")

    # thr if ds
    thr = 0.0
    if htm_version == "htm_ds":
        thr = input("Please insert the threshold: ")
        try:
            thr = float(thr)
        except:
            print(f"Invalid input: {e}")
            return
    
    # ask input from user: he needs to select a topic id
    tr_tpcs = input(f"Please select the number of training topics for the submodel: ")
    try:
        tr_tpcs = int(tr_tpcs)
    except ValueError as e:
        print(f"Invalid input: {e}")
        return
    
    submodel_path = hmg.create_submodel_tr_corpus(
        father_model_path=father_model,
        langs=langs,
        exp_tpc=topic_id,
        tr_topics=tr_tpcs,
        htm_version=htm_version,
        thr=thr)
    
    # train model
    start_time = time.time()
    model = PolylingualTM(
        lang1=langs[0],
        lang2=langs[1],
        model_folder= submodel_path,
        num_topics=tr_tpcs,
        is_second_level=True
    )
    model.train()
    
    end_time = time.time()
    print(f"-- Model trained in {end_time - start_time} seconds")
    
    # Display topics of the submodel 
    # Load topic-keys in each lang
    all_keys = {}
    for lang in langs:
        # Default to white if color is not defined
        color = lang_colors.get(lang, "white")
        print(colored("#" * 50, color))
        print(colored(f"-- -- Topic keys in {lang.upper()}: ", color))
        print(colored("-" * 50, color))
        keys = []
        with (submodel_path / f"mallet_output/keys_{lang}.txt").open('r', encoding='utf8') as fin:
            keys = [el.strip() for el in fin.readlines()]
        all_keys[lang] = keys
        for id, tpc in enumerate(keys):
            print(colored(f"- Topic {id}: {tpc}", color))
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--father_model',
        type=str,
        required=False,
        default="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/POLI_FILTERED_AL/rosie_1_20",
        help="Path to the father model to generate the hierarchy from.")
    args = parser.parse_args()

    main(args.father_model)
