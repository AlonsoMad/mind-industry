import pandas as pd
import deepl
import concurrent.futures

# Load DataFrame
df = pd.read_excel("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/df_q_eval_29jan_topic3_es.xlsx")

# Keep last 1200 rows
df = df.tail(1200)

# Initialize DeepL Translator
auth_key = "56775522-9293-4ab6-b06d-28669192c5b3:fx"  # Replace with your actual API key
translator = deepl.Translator(auth_key)

# Multithreading for faster translations
def translate_text(text):
    try:
        return translator.translate_text(text, source_lang="ES", target_lang="EN-US").text
    except Exception as e:
        return f"Error: {e}"

# Filter out NaNs
df['all_results_content'] = df['all_results_content'].fillna('')

# Perform translation in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    df['all_results_content_tr'] = list(executor.map(translate_text, df['all_results_content'].astype(str)))

# Save translated DataFrame
output_path = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/df_q_eval_29jan_topic3_es_translated_1200.xlsx"
df.to_excel(output_path, index=False)

print(f"Translation completed! Saved to {output_path}")
