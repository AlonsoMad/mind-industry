import dspy
import os
from dotenv import load_dotenv
import pathlib
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFinetune
import dsp
import pandas as pd
from sklearn.model_selection import train_test_split

###########
# API KEY #
###########
path_env = pathlib.Path(os.getcwd()).parent.parent.parent / '.env'
print(path_env)
load_dotenv(path_env)
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

###########
#   LLM   #
###########
llm = dspy.OpenAI(model= "gpt-3.5-turbo", max_tokens=1000)
dspy.settings.configure(lm=llm)

class QAChecker(dspy.Signature):
    """Evaluate the relationship between two ANSWER1 and ANSWER2 to the given QUESTION.
    """
    
    QUESTION = dspy.InputField(desc="The question for which the answers are being evaluated.")
    ANSWER1 = dspy.InputField(desc="The first answer to the question, used as the reference point.")
    ANSWER2 = dspy.InputField(desc="The second answer to the question, evaluated against the first answer.")
    LABEL = dspy.OutputField(desc="The relationship of ANSWER2 to ANSWER1: CONSISTENT or CONTRADICTORY")
    RATIONALE = dspy.OutputField(desc="The explanation or justification for the assigned LABEL.")
    
    
class LinQAChecker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.checker = dspy.ChainOfThought(QAChecker)
    
    def forward(self, question, gold_answer, gen_answer):        
        contrad = None
        if gen_answer:
            try:
                contrad = self.checker(QUESTION=question, ANSWER1=gold_answer, ANSWER2=gen_answer)
            except Exception as e:
                print(f"-- -- Error generating answer: {e}")
                contrad = None
        
        return dspy.Prediction(
            question=question,
            gold_answer=gold_answer,
            pred_answer=gen_answer if gen_answer else None,
            label=self.process_label(contrad.LABEL) if contrad else None,
            rationale=contrad.RATIONALE if contrad else None,
        )
        
    def process_label(self, label_resp):
        norm = dsp.normalize_text(label_resp).lower()
        if "consistent" in norm:
            clean_pred = "CONSISTENT"
        elif "contradictory" in norm or "contradiction" in norm:
            clean_pred = "CONTRADICTION"
        else:
            clean_pred = "FAILED"
        return clean_pred

## CREATE TRAIN AND DEV SET
def create_dtset(path_data="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/contradictions/merged_file.json"):
    # Read the JSON data
    df = pd.read_json(path_data, orient='records', lines=True)

    # Convert the DataFrame into a list of dictionaries
    data = []
    for i, row in df.iterrows():
        data.append({
            "id": i,
            "question": row["question"],
            "gold_answer": row["answer1"],
            "gen_answer": row["answer2"],
            "label": row["label"],
            "rationale": row["rationale"]
        })

    # Convert the list of dictionaries back into a DataFrame
    data_df = pd.DataFrame(data)

    # Split the DataFrame into training and testing sets
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

    # Convert the training and testing DataFrames back into lists of dictionaries
    data_train = train_df.to_dict('records')
    data_test = test_df.to_dict('records')

    # Create training examples
    trainset = [
        dspy.Example({**row}).with_inputs('question', 'gold_answer', 'gen_answer') for row in data_train
    ]

    # Create development examples
    devset = [
        dspy.Example({**row}).with_inputs('question', 'gold_answer', 'gen_answer') for row in data_test
    ]

    return trainset, devset
    
def validate_answer_contained(example, pred, trace=None):
    print(pred.label, example.label, pred.label == example.label)
    return pred.label == example.label

teleprompter = BootstrapFewShotWithRandomSearch(
    metric=validate_answer_contained,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    num_candidate_programs=16,
    max_rounds=1,
)

trainset, devset = create_dtset()
compiled_classifier = teleprompter.compile(LinQAChecker(), trainset=trainset, valset=devset)

compiled_classifier.save("LinQAChecker-saved.json")

import pdb; pdb.set_trace()