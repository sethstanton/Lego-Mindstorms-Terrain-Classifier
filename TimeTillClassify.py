import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def load_and_create_feature_matrix(file_name, label, sensor_columns, rows_per_set):
    data = []
    current_chunk = []

    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith('-----'):
                if current_chunk:
                    data.extend(current_chunk[:rows_per_set])
                    current_chunk = []
            elif line.strip() and '[' in line and ']' in line:
                line_data = line.strip().strip('[]').split(', ')
                current_chunk.append([float(val) for val in line_data[4:]])

        if current_chunk:
            data.extend(current_chunk[:rows_per_set])

    df = pd.DataFrame(data, columns=sensor_columns)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[sensor_columns] = scaler.fit_transform(df[sensor_columns])
    df['label'] = label
    return df


sensor_columns = ['Pitch', 'Roll', 'AccX', 'AccY', 'AccZ', 'AngVelPitch', 'AngVelRoll', 'AngVelYaw', 'DownSensor', 'FrontSensor']
subdirectories = ['TimeTillClassify/Forward', 'TimeTillClassify/Left', 'TimeTillClassify/Right']
surfaces = ['Carpet', 'Cardboard', 'Stones', 'Rice', 'Wood']
rows_per_set = 25
# 50 = 15sec, 33 = 10 seconds, 25 = 7.5sec , 17 = 5 seconds, 8 = 2.5 seconds,  3 = 1 seconds

all_data_frames = []
for subdir in subdirectories:
    for surface in surfaces:
        file_name = f'{subdir}/{surface}.txt'
        df = load_and_create_feature_matrix(file_name, surface, sensor_columns, rows_per_set)
        all_data_frames.append(df)

features_df = pd.concat(all_data_frames, ignore_index=True)
le = LabelEncoder()
y_encoded = le.fit_transform(features_df['label'])

X = features_df.drop('label', axis=1)
y = y_encoded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
feature_importance_values = np.zeros(len(X_train.columns))

for train_idx, test_idx in cv.split(X_train, y_train):
    X_cv_train, X_cv_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_cv_train, y_cv_test = y_train[train_idx], y_train[test_idx]
    rf.fit(X_cv_train, y_cv_train)
    feature_importance_values += rf.feature_importances_

feature_importance_values /= 5
sorted_idx = np.argsort(feature_importance_values)[::-1]
top5_features = X_train.columns[sorted_idx[:3]]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances from Cross-Validated RandomForest")
plt.bar(X_train.columns[sorted_idx], feature_importance_values[sorted_idx], color='r')
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

print(f"The top 5 features are: {top5_features.tolist()}")

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
    conf_matrix, class_report = evaluate_model(model, X_train[top5_features], y_train, X_test[top5_features], y_test, name, cv)
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
