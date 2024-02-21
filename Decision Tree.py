import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

def load_data_part(dataset_path, part, total_parts):
    try:
        chunk_size = int(1 / total_parts * len(pd.read_csv(dataset_path)))
        chunks = pd.read_csv(dataset_path, chunksize=chunk_size)
        for i, chunk in enumerate(chunks):
            if i == part - 1:
                return chunk
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        exit()
    except pd.errors.EmptyDataError:
        print("The file is empty.")
        exit()

dataset_path = input("Please enter the path to your dataset file (CSV format): ")
num_iterations = int(input("Enter the number of iterations: "))

precision_scores = []
recall_scores = []
f1_scores = []

for i in range(num_iterations):
    data_part = load_data_part(dataset_path, i + 1, num_iterations)
    X_data = data_part.iloc[:, :-1]  
    y_data = data_part.iloc[:, -1]   

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_data_encoded = encoder.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data_encoded, y_data, test_size=0.2, random_state=42)

    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    dt_predictions = dt_classifier.predict(X_test)

    print(f"\nIteration {i+1} Results:")
    print("Accuracy:", accuracy_score(y_test, dt_predictions))
    print("Classification Report:\n", classification_report(y_test, dt_predictions))

    precision_scores.append(precision_score(y_test, dt_predictions, average='weighted', zero_division=1))
    recall_scores.append(recall_score(y_test, dt_predictions, average='weighted', zero_division=1))
    f1_scores.append(f1_score(y_test, dt_predictions, average='weighted', zero_division=1))

print("\nAverage Precision, Recall, and F1-score with zero_division=1 for Decision Tree over iterations:")
print("Decision Tree Average Precision:", np.mean(precision_scores))
print("Decision Tree Average Recall:", np.mean(recall_scores))
print("Decision Tree Average F1-score:", np.mean(f1_scores))
