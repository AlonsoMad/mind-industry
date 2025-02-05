import pathlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import shuffle
import pandas as pd
import scipy.sparse as sparse
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Read labels
path_labels = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/topic_modeling/labels/annotations_en.xlsx"
path_predict = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/topic_modeling/labels/all_en.csv"

# Read training labels
print("Reading labels...")
df = pd.read_excel(path_labels)
print("Labels shape:", df.shape)

# Read test data
print("Reading test data...")
df_predict = pd.read_csv(path_predict)
print("Test data shape:", df_predict.shape)

# Remove annotated documents from the test data
df_predict = df_predict[~df_predict["doc_id"].isin(df["doc_id"])]

# Parse the thetas column
def parse_thetas(row):
    return np.array([float(x) for x in row.strip("[]").split()])

# Training data
texts_train = df.lemmas_x.values.tolist()
labels_train = df.label.values.tolist()
texts_train_no_lemmas = df.text.values.tolist()
thetas_train = np.array(df["thetas"].apply(parse_thetas).tolist())
outside_matrix_train = sparse.csr_matrix(thetas_train)
scores_train = sparse.csr_matrix(df.doc_score.values.reshape(-1, 1))

# Test data
texts_test = df_predict.lemmas_x.values.tolist()
texts_test_no_lemmas = df_predict.text.values.tolist()
thetas_test = np.array(df_predict["thetas"].apply(parse_thetas).tolist())
outside_matrix_test = sparse.csr_matrix(thetas_test)
scores_test = sparse.csr_matrix(df_predict.doc_score.values.reshape(-1, 1))

# Shuffle the training data
texts_train, labels_train, outside_matrix_train, scores_train, texts_train_no_lemmas = shuffle(
    texts_train, labels_train, outside_matrix_train, scores_train, texts_train_no_lemmas, random_state=42
)

# TF-IDF Vectorizer
print("Vectorizing text data...")
tfidf = TfidfVectorizer()

# Transform the text data
X_train_tfidf = tfidf.fit_transform(texts_train)
X_test_tfidf = tfidf.transform(texts_test)

print("TF-IDF shape:", X_train_tfidf.shape)

# Concatenate sparse matrices: TF-IDF features and outside matrix
X_train = sparse.hstack([X_train_tfidf, outside_matrix_train])
X_test = sparse.hstack([X_test_tfidf, outside_matrix_test])

print("Final train shape:", X_train.shape)
print("Final test shape:", X_test.shape)

# Define the SVM model
svm = SVC(probability=True)

# Define the parameter grid for optimization
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=2)

# Train the model
print("Training the model...")
grid_search.fit(X_train, labels_train)

# Best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Evaluate on the test set
best_model = grid_search.best_estimator_

# Get probabilities for the test set
probabilities = best_model.predict_proba(X_test)

# Extract the probabilities for the positive class
positive_class_probabilities = probabilities[:, 1]

# Define the custom threshold for positive classification
threshold = 0.9

# Apply the threshold
custom_predictions = (positive_class_probabilities > threshold).astype(int)

# Save predictions along with the probabilities and text
test_df = pd.DataFrame({
    'lemmas': texts_test,
    'text': texts_test_no_lemmas,
    'predicted_probability': positive_class_probabilities,
    'predicted_label': custom_predictions
})
import pdb; pdb.set_trace()