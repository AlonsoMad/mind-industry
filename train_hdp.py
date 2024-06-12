import pandas as pd
import tomotopy as tp

# Load your dataframe
# Assume the dataframe is named 'df' and the column with the text is named 'text'
df = pd.read_parquet("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/translated/df_1.parquet")
df = df[df.lang == "EN"]

# Create a HDP model
hdp_model = tp.HDPModel()

# Add documents to the HDP model
print("Adding documents to the model...")
[hdp_model.add_doc(doc.split()) for doc in df['lemmas']]    

# Train the HDP model
iterations = 1000
hdp_model.train(iter=iterations)

# Get the final number of topics
import pdb; pdb.set_trace()
final_num_topics = hdp_model.num_topics
print(f"Final number of topics: {final_num_topics}")

# Print the results
for i in range(hdp_model.k):
    print(f"Topic #{i}")
[print(hdp_model.get_topic_words(i, top_n=10)) for i in range(hdp_model.k)]

# Save the model
hdp_model.save('hdp_model.bin', True)

print("Model training complete and model saved.")#58
