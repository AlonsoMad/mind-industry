import pandas as pd

file_path_en = "/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/pltm/en.txt"

file_path_de = "/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/pltm/de.txt"

with open(file_path_en, "r") as file:
    document = file.read()
lines = document.split("\n") 
pairs_en = []  
for line in lines:
    if "\tEN\t" in line: 
        parts = line.split("\tEN\t")  
        if len(parts) == 2:  
            before_en, after_en = parts  
            pairs_en.append((before_en, after_en, "EN"))  
            
with open(file_path_de, "r") as file:
    document = file.read()
lines = document.split("\n") 
pairs_de = []  
for line in lines:
    if "\tDE\t" in line:  
        parts = line.split("\tDE\t") 
        if len(parts) == 2:  
            before_de, after_de = parts 
            pairs_de.append((before_de, after_de, "DE")) 

# Create a DataFrame from the pairs list
df = pd.DataFrame(pairs_en + pairs_de, columns=["doc_id", "text", "lang"])

print(df)

df.to_parquet("/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/pltm/df.parquet")
