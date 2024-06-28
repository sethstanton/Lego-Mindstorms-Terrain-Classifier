import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

"""
This code loads sensor data to extract features, apply machine learning models, and evaluate their performance. 
It includes feature scaling, model training with cross-validation, and performance evaluation through various metrics.
Confusion matrices are generated to help with analysis.
"""

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

def load_and_create_feature_matrix(file_name, label, sensor_columns):
    """
    Reads and processes sensor data from a text file, applies min max scaling, and returns a DataFrame.
    Inputs:
           file_name: String, path to the sensor data file.
           label: String, category label for the data.
           sensor_columns: List of strings, specifies the columns to be processed.
    Output:
           df: DataFrame containing scaled features and their corresponding label."""
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            if line.strip() and '[' in line and ']' in line:
                line_data = line.strip().strip('[]').split(', ')
                data.append([float(val) for val in line_data[4:]])
    df = pd.DataFrame(data, columns=sensor_columns)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[sensor_columns] = scaler.fit_transform(df[sensor_columns])
    df['label'] = label
    return df

sensor_columns = ['Pitch', 'Roll', 'AccX', 'AccY', 'AccZ', 'AngVelPitch', 'AngVelRoll', 'AngVelYaw', 'DownSensor', 'FrontSensor']
subdirectories = ['SecondRound/Forward', 'SecondRound/Left', 'SecondRound/Right']
surfaces = ['Carpet', 'Cardboard', 'Stones', 'Rice', 'Wood']

all_data_frames = []
for subdir in subdirectories:
    for surface in surfaces:
        file_name = f'{subdir}/{surface}.txt'
        df = load_and_create_feature_matrix(file_name, surface, sensor_columns)
        all_data_frames.append(df)

features_df = pd.concat(all_data_frames, ignore_index=True)
le = LabelEncoder()
y_encoded = le.fit_transform(features_df['label'])

X = features_df.drop('label', axis=1)
y = y_encoded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVC': SVC(kernel='linear', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'LGBM': LGBMClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro'),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, cv):
    """
Fits the model, performs cross-validation, and prints out the classification metrics and confusion matrix.
Inputs:
        model: The machine learning model to be used.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        model_name: String, name of the model.
        cv: Cross-validation splitting strategy.
Outputs:
       conf_matrix: Confusion matrix of the test predictions.
       class_report: Classification report showing main classification metrics.
    """
    print(f"\nEvaluating {model_name}...")
    model.fit(X_train, y_train)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=le.classes_)

    print(f"Results for {model_name}:")
    print(f"Accuracy: {np.mean(scores['test_accuracy']):.2f} ± {np.std(scores['test_accuracy']):.2f}")
    print(f"Precision: {np.mean(scores['test_precision']):.2f}")
    print(f"Recall: {np.mean(scores['test_recall']):.2f}")
    print(f"F1 Score: {np.mean(scores['test_f1']):.2f}")
    print(f"ROC AUC: {np.mean(scores['test_roc_auc']):.2f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    for class_name, accuracy in zip(le.classes_, class_accuracies):
        print(f"Accuracy for {class_name}: {accuracy:.2f}")

    return conf_matrix, class_report

"""
Here we have two loops that build the confusion matrixes and class reports before the dictionary of confusion matrixes
are used to create visual versions in matplotlib.
"""
confusion_matrices = {}
classification_reports = {}
for name, model in classifiers.items():
    conf_matrix, class_report = evaluate_model(model, X_train, y_train, X_test, y_test, name, cv)
    confusion_matrices[name] = conf_matrix
    classification_reports[name] = class_report

for name, conf_matrix in confusion_matrices.items():
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(le.classes_))
    plt.xticks(tick_marks, le.classes_, rotation=45)
    plt.yticks(tick_marks, le.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()
