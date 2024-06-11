"""
BLADE: Bad Low Affiliation Document Examiner

BLADE is an active learning classifier designed to identify and flag "bad" documents within a corpus. A document is considered "bad" if its words are assigned to all topics with consistently low probabilities, indicating poor topic representation and a lack of strong affiliation with any particular topic. 

Author: Lorena Calvo-Bartolomé
Date: 24.05.2025
"""

import copy
import logging
import pickle
from pathlib import Path

import numpy as np
from doc_selector import DocSelector
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


class Blade(object):
    def __init__(
        self,
        model_path: str = None,
        lang: str = None,
        source_path: str = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/df_1.parquet",
        state_path: str = None,
        metric_classifier: int = 4,
        logger: logging.Logger = None,
        blade_state_path: str = None
    ):
        self._logger = logger if logger else logging.getLogger(__name__)
        
        if blade_state_path and Path(blade_state_path).exists():
            
            self._logger.info(f"-- -- Loading BLADE from {blade_state_path}...")
            with open(blade_state_path, 'rb') as file:
                state = pickle.load(file)
                self.__dict__.update(state.__dict__)
            self._logger.info(f"Blade object loaded from {blade_state_path}")
        else:
            
            self._logger.info(f"-- -- No exisiting BLADE object found. Initializing from scratch...")
            self.doc_selector = DocSelector(
                Path(model_path), Path(source_path), lang=lang, logger=self._logger)

            self.thetas = self.doc_selector.output_lang["thetas"]
            self.S3 = self.doc_selector.invoke_method_by_id(metric_classifier)
            self.df_docs = self.doc_selector.output_lang["df"]

            # Initialize columns for labels and human-labeled flag
            self.df_docs['label'] = np.nan
            self.df_docs['human_labeled'] = False
            if state_path:
                self.state_path = state_path
            else:
                self.state_path = Path(model_path) / f"df_docs_updated_{lang}.csv"

            # Compute TF-IDF features
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_features = tfidf_vectorizer.fit_transform(
                self.df_docs['lemmas_x_x'])

            # Combine thetas, S3, and TF-IDF features i
            self.X = hstack(
                [csr_matrix(copy.deepcopy(self.thetas)).astype(np.float64),
                 csr_matrix(copy.deepcopy(self.S3)).astype(np.float64),
                 tfidf_features
                 ], format='csr'
            )

            try:
                with open("words.txt", "r") as file:
                    self.keys = [line.strip() for line in file]
            except:
                self.keys = ['infant', 'postpartum', 'pregnant']

            self._init_classifier()

            self.X_train = np.empty((0, self.X.shape[1]))
            self.y_train = np.array([])

            self.X_pool = self.X
            self.df_pool = self.df_docs.copy()

            self._preprocess_indices()
            self.original_to_current_index = {
                i: i for i in range(len(self.df_pool))}

            self.save(blade_state_path)

    def _init_classifier(self):
        self.learner = SGDClassifier(
            loss="log_loss", penalty='l2', tol=1e-3, random_state=42, learning_rate="optimal", eta0=0.1, validation_fraction=0.2, alpha=0.000005)
        self._logger.info("-- -- Active Learner initialized.")

    def _preprocess_indices(self):
        self.positive_indices = self.df_pool[self.df_pool['text'].str.contains(
            '|'.join(self.keys), case=False, na=False)].index.to_list()
        self.avg_S3 = np.mean(self.S3, axis=1)
        self.negative_indices = np.argsort(self.avg_S3).tolist()

    def update_indices(self, used_index):
        if used_index in self.positive_indices:
            self.positive_indices.remove(used_index)
        if used_index in self.negative_indices:
            self.negative_indices.remove(used_index)

    def preference_function(self, iteration):
        if len(self.y_train) > 0:
            probas = self.learner.predict_proba(self.X_pool)
            uncertainty = -np.sum(probas * np.log(probas + 1e-10), axis=1)
        else:
            uncertainty = np.ones(self.X_pool.shape[0])

        selection_type = iteration % 3

        if selection_type == 0 and len(self.positive_indices) > 0:
            valid_positive_indices = [self.original_to_current_index[i]
                                      for i in self.positive_indices if i in self.original_to_current_index]
            if valid_positive_indices:
                positive_uncertainty = uncertainty[valid_positive_indices]
                selected_idx = valid_positive_indices[np.argmax(
                    positive_uncertainty)]
            else:
                selected_idx = np.argmax(uncertainty)

        elif selection_type == 1 and len(self.negative_indices) > 0:
            valid_negative_indices = [self.original_to_current_index[i]
                                      for i in self.negative_indices if i in self.original_to_current_index]
            if valid_negative_indices:
                combined_scores = uncertainty[valid_negative_indices] / \
                    (self.avg_S3[valid_negative_indices] + 1e-10)
                selected_idx = valid_negative_indices[np.argmax(
                    combined_scores)]
            else:
                selected_idx = np.argmax(uncertainty)

        else:
            selected_idx = np.argmax(uncertainty)

        return [selected_idx]

    def request_labels(self, query_instances, indices):
        labels = []
        for query_instance, idx in zip(query_instances, indices):
            doc_id = self.df_pool.iloc[idx]['id_top']
            doc_content = self.df_pool.iloc[idx]['text']
            print(f"Document ID: {doc_id}")
            print(f"Document Content: {doc_content}")
            label = int(
                input("Please provide the label for the queried instance (0 or 1): "))
            labels.append(label)
        return np.array(labels)

    def active_learning_loop(self, n_queries=10):
        for idx in range(n_queries):
            preferred_indices = self.preference_function(idx)
            query_idx = preferred_indices[0]
            query_instance = self.X_pool[query_idx].reshape(1, -1)
            label = self.request_labels(query_instance, [query_idx])[0]

            # Update training data
            self.X_train = vstack([self.X_train, query_instance])
            self.y_train = np.append(self.y_train, label)

            # Fit the classifier with the new data
            self.learner.partial_fit(
                self.X_train, self.y_train, classes=np.array([0, 1]))

            # Update df_docs with the label
            original_index = list(self.original_to_current_index.keys())[
                list(self.original_to_current_index.values()).index(query_idx)]
            self.df_docs.loc[original_index, 'label'] = label
            self.df_docs.loc[original_index, 'human_labeled'] = True

            # Save the updated DataFrame to a file
            self.df_docs.to_csv(self.state_path, index=False)

            # Remove queried instance from the pool
            self.X_pool = vstack(
                [self.X_pool[:query_idx], self.X_pool[query_idx+1:]])
            self.df_pool = self.df_pool.drop(
                self.df_pool.index[query_idx]).reset_index(drop=True)

            # Update the mapping
            self.original_to_current_index.pop(original_index)
            self.original_to_current_index = {
                orig: cur-1 if cur > query_idx else cur for orig, cur in self.original_to_current_index.items()}

            self.update_indices(original_index)

            self._logger.info(
                f'Iteration {idx + 1}/{n_queries}, Document ID: {query_idx}')

    def predict(self):
        """
        Predict the labels for the remaining pool of documents.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the predicted labels or None if no labeled data is available
        """
        if len(self.y_train) > 0:
            predictions = self.learner.predict(self.X_pool)
            self.df_pool['predicted_label'] = predictions
            self.df_pool['human_labeled'] = False

            # Merge the predictions with the original df_docs
            for idx in self.df_pool.index:
                original_index = list(self.original_to_current_index.keys())[
                    list(self.original_to_current_index.values()).index(idx)]
                self.df_docs.loc[original_index,
                                 'label'] = self.df_pool.loc[idx, 'predicted_label']
                self.df_docs.loc[original_index, 'human_labeled'] = False

            # Save the updated DataFrame to a file
            self.df_docs.to_csv(self.state_path, index=False)

            return self.df_pool[['id_top', 'text', 'predicted_label']]
        else:
            self._logger.info('No labeled data available to train the model.')
            return None

    def save(self, file_path: str):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        self._logger.info(f"Blade object saved to {file_path}")

    @staticmethod
    def load(file_path: str):
        with open(file_path, 'rb') as file:
            blade = pickle.load(file)
        blade._logger.info(f"Blade object loaded from {file_path}")
        return blade
