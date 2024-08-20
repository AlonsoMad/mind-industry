from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

class GenerateFacts(dspy.Signature):
    """
    Extract self-contained and fully contextualized facts from the given passage.    
    """
    
    passage = dspy.InputField(desc="The passage may contain one or several claims")
    facts = dspy.OutputField(desc="List of self-contained and fully contextualized claims in the form 'subject + verb + object' without using pronouns or vague references")
    
class FactsGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_facts = dspy.ChainOfThought(GenerateFacts)

    def process_facts(self, facts):
        nolist = False

        if "Facts:" in facts:
            facts = facts.split("Facts:")[1]
        elif "facts:" in facts:
            facts = facts.split("facts:")[1]

        try:
            facts = contractions.fix(facts)
        except Exception as e:
            print("Could not expand contrad")
            print(e)

        if '’' in facts:
            facts = facts.replace('’', "'")
        if '“' in facts:  # Changed elif to if
            facts = facts.replace('“', "'")
        if '”' in facts:  # Changed elif to if
            facts = facts.replace('”', "'")

        facts = facts.replace('"',"'")

        if "1." in facts:
            try:
                # Process facts, ensuring non-empty lines
                facts = [re.sub(r'^\d+\.\s*', '', fact).replace('"', "'") for fact in facts.split('\n') if fact.strip()]
                return facts
            except Exception as e:
                print("The error is 1")
                print(e)
                print(facts)
                return facts

        
        if facts.startswith("[") and not facts.endswith("]"):
            facts = facts + "]"
        elif not facts.startswith("[") and facts.endswith("]"):
            facts = "[" + facts
        elif not facts.startswith("[") and not facts.endswith("]"):
            nolist = True
            try:
                facts = [el.strip().replace('"',"'") for el in facts.split(".") if len(el) > 1]
                return facts
            except Exception as e:
                print("The error is 2")
                print(e)
                return facts

    
        try:
            facts = [el.strip().replace('"',"'") for el in facts.split(".") if len(el) > 1]#ast.literal_eval(facts)
        except Exception as e:
            print("The error is 3")
            print(e)
            print(facts)
            
        return facts
            
    def forward(self, passage):
        facts = self.generate_facts(passage=passage).facts
        processed_facts = self.process_facts(facts)
        return dspy.Prediction(facts = processed_facts)

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
        train_data = pd.read_csv( pathlib.Path(data_fpath))

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

config = dict(max_bootstrapped_demos=4, max_labeled_demos=16, num_candidate_programs=2, max_rounds=1)
teleprompter = BootstrapFewShotWithRandomSearch(metric=combined_score, **config)

compiled_pred = teleprompter.compile(FactsGenerator(), trainset=trainset, valset=devset)

compiled_pred.save("compiled_fact.json")