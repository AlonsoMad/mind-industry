from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.datasets import Dataset
from typing import Optional, Union
import ast
import contractions
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from dspy.teleprompt import COPRO


################################################################################
# LLM Configuration
################################################################################
llm = dspy.HFClientTGI(model="meta-llama/Meta-Llama-3-8B ", port=8090, url="http://127.0.0.1")
dspy.settings.configure(lm=llm)
################################################################################

class FactsDataset(Dataset):

    def __init__(
        self,
        data_fpath: str,
        dev_size: Optional[float] = 0.2,
        test_size: Optional[float] = 0.2,
        text_key: str = "passage",
        seed: Optional[int] = 11235,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._train = []
        self._dev = []
        self._test = []

        # Read the training data
        train_data = pd.read_csv(pathlib.Path(data_fpath))

        train_data, temp_data = train_test_split(
            train_data, test_size=dev_size + test_size, random_state=seed)
        dev_data, test_data = train_test_split(
            temp_data, test_size=test_size / (dev_size + test_size), random_state=seed)

        self._train = [
            dspy.Example({**row}).with_inputs(text_key) for row in self._convert_to_json(train_data)
        ]
        self._dev = [
            dspy.Example({**row}).with_inputs(text_key) for row in self._convert_to_json(dev_data)
        ]
        self._test = [
            dspy.Example({**row}).with_inputs(text_key) for row in self._convert_to_json(test_data)
        ]

    def _convert_to_json(self, data: pd.DataFrame):
        if data is not None:
            return data.to_dict(orient='records')

class GenerateFacts(dspy.Signature):
    """
    Extract self-contained and fully contextualized facts from the given passage.    
    """

    passage = dspy.InputField(
        desc="The passage may contain one or several claims")
    facts = dspy.OutputField(
        desc="List of self-contained and fully contextualized claims in the form 'subject + verb + object' without using pronouns or vague references", prefix="Facts:")

class FactsGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_facts = dspy.Predict(GenerateFacts)

    def process_facts(self, facts):
            # Normalize and clean the facts string
        if "Facts:" in facts:
            facts = facts.split("Facts:", 1)[1]
        elif "facts:" in facts:
            facts = facts.split("facts:", 1)[1]

        try:
            facts = contractions.fix(facts)
        except Exception as e:
            print("Could not expand contractions:", e)

        # Replace problematic characters
        replacements = {
            '’': "'",
            '“': "'",
            '”': "'",
            '"': "'"
        }
        for old_char, new_char in replacements.items():
            facts = facts.replace(old_char, new_char)

        # Handle numbered list format
        if "1." in facts:
            try:
                facts_list = [re.sub(r'^\d+\.\s*', '', fact).strip()
                            for fact in facts.split('\n') if fact.strip()]
                return facts_list
            except Exception as e:
                print("Error processing numbered list:", e)
                return []

        # Handle cases with missing brackets
        facts = facts.strip()
        if facts and not (facts.startswith("[") and facts.endswith("]")):
            facts = facts.strip('[]')  # Remove any stray brackets
            try:
                facts_list = [fact.strip() for fact in facts.split('.') if fact.strip()]
                return facts_list
            except Exception as e:
                print("Error processing facts:", e)
                return []

        # General fallback processing
        try:
            facts_list = [fact.strip() for fact in facts.split('.') if fact.strip()]
        except Exception as e:
            print("General error processing facts:", e)
            return []

        return facts_list

    def forward(self, passage):
        facts = self.generate_facts(passage=passage).facts
        processed_facts = self.process_facts(facts)
        return dspy.Prediction(facts=processed_facts)

def combined_score(example, pred, trace=None):
    def sbert_similarity_score(example, pred, trace=None):
        try:
            scores = []

            predicted_lst = pred["facts"]
            try:
                gt_lst = ast.literal_eval(example.facts)
            except Exception as e:
                print("Error in parsing ground truth facts: ", e)
                gt_lst = example.facts.split(".")

            min_facts = min(len(predicted_lst), len(gt_lst))

            # Generate embeddings for predicted and ground truth facts
            predicted_embeddings = model.encode(predicted_lst[:min_facts])
            gt_embeddings = model.encode(gt_lst[:min_facts])

            # Calculate cosine similarity for each pair of embeddings
            for pred_emb, gt_emb in zip(predicted_embeddings, gt_embeddings):
                similarity = 1 - cosine(pred_emb, gt_emb)
                scores.append(similarity)

            # Return the average similarity score
            return np.mean(scores)

        except Exception as e:
            print("An error occurred: ", e)
            print("predicted_lst: ", predicted_lst)
            print("gt_lst: ", gt_lst)
            return 0.0

    return sbert_similarity_score(example, pred, trace)


dataset = FactsDataset(data_fpath="facts_gpt4.csv", dev_size=0.1)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

trainset = dataset._train
devset = dataset._dev
testset = dataset._test

config = dict(max_bootstrapped_demos=4, max_labeled_demos=16,
              num_candidate_programs=2, max_rounds=2)
teleprompter = BootstrapFewShotWithRandomSearch(
    metric=combined_score, **config)

#teleprompter = COPRO(
#    metric=combined_score,
#    verbose=True,
#    depth=10
#    breadth=2,
#)

#kwargs = dict(num_threads=64, display_progress=True, display_table=0) # Used in Evaluate class in the optimization process

#compiled_prompt_opt = teleprompter.compile(FactsGenerator(), trainset=devset,eval_kwargs=kwargs)

compiled_pred = teleprompter.compile(
    FactsGenerator(), trainset=trainset, valset=devset)

compiled_pred.save("compiled_fact.json")
import pdb; pdb.set_trace()