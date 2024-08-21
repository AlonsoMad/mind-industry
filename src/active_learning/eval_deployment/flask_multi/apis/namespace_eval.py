import pathlib
from flask_restx import Namespace, Resource, reqparse
from flask import jsonify
import logging
import pandas as pd
import os

# Set up logging
logger = logging.getLogger(__name__)

# Get paths from environment variables
annotations_path_en = os.environ.get('ANNOTATION_PATH_EN', '/default/path/to/model')
annotations_path_es = os.environ.get('ANNOTATION_PATH_ES', '/default/path/to/source')
annotations_path_second = os.environ.get('ANNOTATION_PATH_SECOND', '/default/path/to/source')

# Load the dataframes at the start
def get_balanced_sample(path, size=0.6):
    # Read the annotations
    df = pd.read_csv(path)
    df = df[df.human_labeled == True]
    sixty_percent_size = int(len(df) * size)

    # Split the dataframe based on labels
    #positive_df = df[df['label'] == 1.0]
    #negative_df = df[df['label'] == 0.0]

    # Determine the number of samples needed from each label to maintain balance
    n_samples_per_label = int(sixty_percent_size // 2)

    #positive_sample = positive_df.sample(n=n_samples_per_label, random_state=1)
    #negative_sample = negative_df.sample(n=n_samples_per_label, random_state=1)

    # Combine the sampled dataframes into one balanced dataframe
    #balanced_sample = pd.concat([positive_sample, negative_sample])

    # Shuffle the combined dataframe
    #balanced_sample = balanced_sample.sample(frac=1, random_state=1).reset_index(drop=True)
    
    sample = df.sample(n=sixty_percent_size, random_state=1)

    return sample

# Get balanced samples for English and Spanish docs
df_en = get_balanced_sample(annotations_path_en, size=0.48366013071895425)
df_es = get_balanced_sample(annotations_path_es, size=1)

# Combine and shuffle the balanced samples
combined_df = pd.concat([df_en, df_es]).sample(frac=1, random_state=1).reset_index(drop=True)
combined_df = combined_df.dropna(subset=['label'])
#combined_df = pd.read_csv(annotations_path_second)
combined_df['third_eval_label'] = None  # Initialize the new column

# Set up a set to keep track of labeled document IDs
labeled_ids = set()

# Set up the Flask-RESTX namespace and parsers
api = Namespace('AL', title='Blade Annotation Eval API')

parser = reqparse.RequestParser()
parser.add_argument('label', help='Label of the document', required=True)
parser.add_argument('idx', help='Index of the document', required=True)
parser2 = reqparse.RequestParser()
parser2.add_argument('idx', help='Index of the document', required=True)

# Define the GetDocumentEval endpoint
@api.route('/GetDocumentEval/')
class GetDocumentEval(Resource):
    @api.doc(parser=parser2)
    def post(self):
        args = parser2.parse_args()
        idx = args['idx']
        
        try:
            if idx in labeled_ids:
                return jsonify({"error": "This document has already been labeled."}), 400
            
            document = combined_df[combined_df.doc_id == idx].to_dict(orient='records')[0]
            if not document:
                raise IndexError(f"Index {idx} is out of range.")
            
            # Logging fetched document
            logger.info(f"Fetched document: {document}")
            
            return jsonify(document)
        
        except IndexError as e:
            logger.error(f"Index {idx} is out of range: {str(e)}")
            return jsonify({"error": "Index out of range."}), 400
        
        except Exception as e:
            logger.error(f"Error fetching document: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

# Define the LabelDocument endpoint
@api.route('/LabelDocument/')
class LabelDocument(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        label = args['label']
        idx = args['idx']
        logger.info(f"Labeling document at index {idx} with label {label}")
        try:
            if idx in labeled_ids:
                return jsonify({"error": "This document has already been labeled."}), 400
            combined_df.loc[combined_df.doc_id == idx, 'third_eval_label'] = label
            labeled_ids.add(idx)  # Add the document ID to the set
            return jsonify({"message": "Document labeled successfully"})
        except Exception as e:
            logger.error(f"Error labeling document: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
        
# Define the LabelDocument endpoint
@api.route('/GetIdDocsToLabel/')
class GetIdDocsToLabel(Resource):
    def post(self):
        logger.info(f"Getting all ids to label")
        try:
            return jsonify({"docs": combined_df.doc_id.unique().tolist()})
        except Exception as e:
            logger.error(f"Error getting document IDs: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
        
@api.route('/SaveState/')
class SaveState(Resource):
    def post(self):
        logger.info("Saving annotations")
        try:
            # Save the annotations to a CSV file
            path_save = pathlib.Path(annotations_path_en).parent / 'second_eval_annotations_upt.csv'
            combined_df.to_csv(path_save, index=False)
            
            return jsonify({"message": "Annotations saved successfully"})
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500