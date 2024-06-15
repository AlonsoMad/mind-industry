from flask_restx import Namespace, Resource, reqparse
from flask import jsonify
import logging
from blade import Blade
import os

# Set up logging
logger = logging.getLogger(__name__)

# Get paths from environment variables
model_path = os.environ.get('MODEL_PATH', '/default/path/to/model')
source_path = os.environ.get('SOURCE_PATH', '/default/path/to/source')
blade_state_path = os.environ.get('BLADE_STATE_PATH', '/default/path/to/blade_state')
lang = os.environ.get('LANG', 'EN')

# Create Blade object
blade = Blade(
    model_path=model_path,
    source_path=source_path,
    lang=lang,
    blade_state_path=blade_state_path
)

api = Namespace('AL', title='Active Learning API')

parser = reqparse.RequestParser()
parser.add_argument('label', help='Label of the document', required=True)

parser2 = reqparse.RequestParser()
parser2.add_argument('idx', help='Index of the document', required=True)

@api.route('/getDocumentToLabel/')
class GetDocumentToLabel(Resource):
    @api.doc(parser=parser2)
    def post(self):
        args = parser2.parse_args()
        idx = int(args['idx'])
        logger.info(f"Fetching document with index {idx}")
        try:
            document = blade.get_document(idx)
            logger.info(f"Document: {document}")
            return jsonify(document)
        except Exception as e:
            logger.error(f"Error fetching document: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

@api.route('/LabelDocument/')
class LabelDocument(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        label = int(args['label'])
        logger.info(f"Labeling document with label {label}")
        try:
            blade.do_update(label)
            return jsonify({"message": "Document labeled successfully"})
        except Exception as e:
            logger.error(f"Error labeling document: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

@api.route('/SaveState/')
class SaveState(Resource):
    def post(self):
        logger.info("Saving blade state")
        try:
            blade.save_state()
            return jsonify({"message": "State saved successfully"})
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500