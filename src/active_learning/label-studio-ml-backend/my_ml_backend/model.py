import logging
import os
import pickle
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from src.active_learning.blade import Blade

logger = logging.getLogger(__name__)

_model: Blade = None

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    ########################################
    # MODEL PARAMETERS
    ########################################
    # Label Studio host - to be used for training
    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://kumo01:9091')
    # Label Studio API key - to be used for training
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    # Start training each N updates
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    MODEL_DIR = os.getenv('MODEL_DIR', '.')
    TM_MODEL_DIR = os.getenv('TM_MODEL_DIR', '.')

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')
        
    def _get_tm_model_features(self):
        
        if not os.path.exists(self.TM_MODEL_DIR):
            logger.error(f'-- -- TM model directory does not exist: {self.TM_MODEL_DIR}')
            return None
        
        # Load the TM model features
        # TODO
        thetas = np.load(os.path.join(self.TM_MODEL_DIR, 'thetas.npy'))
        S3 = np.load(os.path.join(self.TM_MODEL_DIR, 'S3.npy'))
        df = pd.read_csv(os.path.join(self.TM_MODEL_DIR, 'df.csv'))
        
        return {"thetas": thetas, "S3": S3, "df": df}
        
    def get_model(self, blank=False):
        
        global _model
        # Lazy initialization of the model
        # If the model is not already initialized, it is initialized here
        if _model is not None:
            logger.debug(f'-- -- Model is already initialized')
            return _model
        else:
            model_path = os.path.join(self.MODEL_DIR, 'model.pkl')
            if not os.path.exists(model_path) or blank:
                features = self._get_tm_model_features()
                _model = Blade(
                    **features
                )
                logger.info(f'-- -- Creating a new model with features loaded from {self.TM_MODEL_DIR}')
                
            else:
                logger.info(f'-- -- Loading model from {model_path}')
                with open(model_path, 'rb') as f:
                    _model = pickle.load(f)
                
            return _model

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]
        
        return ModelResponse(predictions=[])
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

