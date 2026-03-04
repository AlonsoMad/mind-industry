import pandas as pd
import pathlib
import os

# 1. Load the generated dataset
input_path = "Tech_EN_ES_temp/dataset"
if not os.path.exists(input_path):
    print(f"Error: {input_path} not found. Run 'python3 generate_tech_dataset.py' first.")
    exit(1)

df = pd.read_parquet(input_path)

# 2. Add required pipeline columns
# Fake lemmas for this test: just copy text
df["lemmas"] = df["text"]
df["lemmas_tr"] = ""  # DataPreparer will fill this if correctly paired
df["full_doc"] = df["title"]
df["doc_id"] = df["lang"] + "_" + df["id_preproc"].astype(str)

# 3. Split into EN and ES
df_en = df[df["lang"] == "EN"].copy()
df_es = df[df["lang"] == "ES"].copy()

# 4. Save to a format data prepare expects
output_dir = pathlib.Path("data/test_tech")
output_dir.mkdir(parents=True, exist_ok=True)

df_en.to_parquet(output_dir / "tech_en.parquet")
df_es.to_parquet(output_dir / "tech_es.parquet")

print(f"Split tech dataset saved to {output_dir}")
print(f"EN: {len(df_en)} rows")
print(f"ES: {len(df_es)} rows")
