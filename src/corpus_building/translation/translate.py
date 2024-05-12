import time
from datasets import load_dataset
from transformers import pipeline as HF_pipeline
from transformers.pipelines.pt_utils import KeyDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import gc
import warnings
warnings.filterwarnings("ignore")

def main():
    
    
    torch.cuda.empty_cache()
    
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    
    dtset = load_dataset(
        "parquet",
        data_files="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.0_en_compiled_passages_lang.parquet",
        split="train"
    )
    lang = "en"
    target = "es"
    small_dataset = dtset.select(range(100)) #25%
    
    
    def tr_pipeline(text):
        with torch.no_grad():
 
            text = f"<2es> {text}"
            model_name = 'jbochi/madlad400-3b-mt'
            model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
            tokenizer = T5Tokenizer.from_pretrained(model_name)

            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
            outputs = model.generate(input_ids=input_ids) 
        
            return {'tr_passage': tokenizer.decode(outputs[0], skip_special_tokens=True)}
    
    start_time = time.time()
    
    tr_dtset = small_dataset.map(tr_pipeline, batched=True)
        
    print(f"Time elapsed: {time.time() - start_time}")
    
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()