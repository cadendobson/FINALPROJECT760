from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from lime.lime_text import LimeTextExplainer
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

nlp = spacy.load("en_core_web_sm")

def LoadData(filename):
    headlines, labels = [], []
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            text = lines[i].strip()
            if i + 1 < len(lines):
                label = lines[i + 1].strip()
                if label in {"0", "1"}:
                    headlines.append(text)
                    labels.append(int(label))
    return headlines, labels

print("Loading train and test datasets...")
train_texts, train_labels = LoadData("train.txt")
test_texts, test_labels = LoadData("test.txt")

def LinguisticFeaturesPull(headlines):
    features = []
    for text in headlines:
        doc = nlp(text)
        tokens = [token.text for token in doc]
        features.append([
            len(doc),
            len(set(tokens)),
            np.mean([len(t) for t in tokens]) if len(tokens) > 0 else 0,
            len(list(doc.sents)),
            len(doc) / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0,
            len([t for t in doc if t.pos_ == "NOUN"]),
            len([t for t in doc if t.pos_ == "VERB"]),
            len([t for t in doc if t.pos_ == "ADJ"]),
            len([t for t in doc if t.pos_ == "ADV"]),
            len(doc.ents)
        ])
    return np.array(features)

print("Extracting linguistic features...")
train_linguistic_features = LinguisticFeaturesPull(train_texts)
test_linguistic_features = LinguisticFeaturesPull(test_texts)

scaler = MinMaxScaler()
train_linguistic_features = scaler.fit_transform(train_linguistic_features)
test_linguistic_features = scaler.transform(test_linguistic_features)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(train_texts)
X_test_vec = vectorizer.transform(test_texts)

print("Applying Chi-Squared feature selection...")
chi2_selector = SelectKBest(chi2, k=1000)
X_train_selected = chi2_selector.fit_transform(X_train_vec, train_labels)
X_test_selected = chi2_selector.transform(X_test_vec)
feature_scores = chi2_selector.scores_
feature_names = vectorizer.get_feature_names_out()

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Score": feature_scores
}).sort_values(by="Score", ascending=False)

print("\nTop 20 features based on Chi-Squared test:")
print(feature_importance.head(20).to_string(index=False))

X_train_combined = np.hstack([X_train_selected.toarray(), train_linguistic_features])
X_test_combined = np.hstack([X_test_selected.toarray(), test_linguistic_features])
y_train = train_labels
y_test = test_labels


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": CalibratedClassifierCV(estimator=LinearSVC(), cv=5),
    "Random Forest": RandomForestClassifier(random_state=2)
}

param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    },
    "Naive Bayes": {
        "alpha": [0.01, 0.1, 1, 5, 10]
    },
    "SVM": {
        "estimator__C": [0.01, 0.1, 1, 10] 
    },
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

class_names = ["LLM", "Human"]
explainer = LimeTextExplainer(class_names=class_names)

def LimeProbabilities(headlines, model):
    vec = vectorizer.transform(headlines)
    vec_sel = chi2_selector.transform(vec)
    ling = LinguisticFeaturesPull(headlines)
    ling = scaler.transform(ling)
    combined = np.hstack([vec_sel.toarray(), ling])
    return model.predict_proba(combined)

for name, model in models.items():
    print(f"\nTuning hyperparameters for {name}...")
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_combined, y_train)
    best_model = grid.best_estimator_

    print(f"Best parameters for {name}: {grid.best_params_}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

    preds = best_model.predict(X_test_combined)
    print(f"\n{name} Test Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test, preds)) if true != pred]
    print(f"Number of misclassifications for {name}: {len(misclassified_indices)}")

    for i, idx in enumerate(misclassified_indices[:1]):
        text = test_texts[idx]
        true_label = y_test[idx]
        predicted_label = preds[idx]

        explanation = explainer.explain_instance(
            text,
            lambda x: LimeProbabilities(x, best_model),
            num_features=10
        )
        print(f"\nMisclassification {i + 1}:")
        print(f"Text: {text}")
        print(f"True Label: {true_label} ({'Human' if true_label == 1 else 'LLM'})")
        print(f"Predicted Label: {predicted_label} ({'Human' if predicted_label == 1 else 'LLM'})")
        filename = f"{name.replace(' ', '_')}_misclassification_{i + 1}.html"
        explanation.save_to_file(filename)
        print(f"LIME explanation saved to {filename}")
