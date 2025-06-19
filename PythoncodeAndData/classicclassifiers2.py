from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load("en_core_web_sm")

def load_dataset(filename, label):
    texts = []
    labels = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            text = lines[i].strip()
            if text:
                texts.append(text)
                labels.append(label)
    return texts, labels

print("Loading datasets...")
human_texts, human_labels = load_dataset("NewsSA.txt", 0)
gpt3_texts, gpt3_labels = load_dataset("sesotho_llm_headlinesCGPT3.5.txt", 1)
gpt4_texts, gpt4_labels = load_dataset("sesotho_llm_headlinesCGPT4.0.txt", 2)

texts = human_texts + gpt3_texts + gpt4_texts
labels = human_labels + gpt3_labels + gpt4_labels

def extract_linguistic_features(texts):
    features = []
    for text in texts:
        doc = nlp(text)
        num_tokens = len(doc)
        num_unique_tokens = len(set([token.text for token in doc]))
        avg_token_length = np.mean([len(token.text) for token in doc]) if len(doc) > 0 else 0
        num_sentences = len(list(doc.sents))
        avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else 0
        num_nouns = len([token for token in doc if token.pos_ == "NOUN"])
        num_verbs = len([token for token in doc if token.pos_ == "VERB"])
        num_adjectives = len([token for token in doc if token.pos_ == "ADJ"])
        num_adverbs = len([token for token in doc if token.pos_ == "ADV"])
        num_named_entities = len(doc.ents)

        features.append([
            num_tokens, num_unique_tokens, avg_token_length,
            num_sentences, avg_sentence_length,
            num_nouns, num_verbs, num_adjectives, num_adverbs,
            num_named_entities
        ])
    return np.array(features)

print("Extracting linguistic features...")
raw_features = extract_linguistic_features(texts)
raw_feature_names = [
    "Num Tokens", "Num Unique Tokens", "Avg Token Length",
    "Num Sentences", "Avg Sentence Length",
    "Num Nouns", "Num Verbs", "Num Adjectives", "Num Adverbs",
    "Num Named Entities"
]
# caden pls fix this
human_raw = raw_features[np.array(labels) == 0]
gpt3_raw = raw_features[np.array(labels) == 1]
gpt4_raw = raw_features[np.array(labels) == 2]

human_means_raw = np.mean(human_raw, axis=0)
gpt3_means_raw = np.mean(gpt3_raw, axis=0)
gpt4_means_raw = np.mean(gpt4_raw, axis=0)

print("Raw Linguistic Feature Averages (Before Scaling):\n")
print(f"{'Feature':<25} {'Human':>10} {'GPT-3.5':>10} {'GPT-4.0':>10}")
print("-" * 60)
for i, name in enumerate(raw_feature_names):
    print(f"{name:<25} {human_means_raw[i]:>10.2f} {gpt3_means_raw[i]:>10.2f} {gpt4_means_raw[i]:>10.2f}")
scaler = MinMaxScaler()
linguistic_features = scaler.fit_transform(raw_features)

feature_names = raw_feature_names

human_features = linguistic_features[np.array(labels) == 0]
gpt3_features = linguistic_features[np.array(labels) == 1]
gpt4_features = linguistic_features[np.array(labels) == 2]

human_means = np.mean(human_features, axis=0)
gpt3_means = np.mean(gpt3_features, axis=0)
gpt4_means = np.mean(gpt4_features, axis=0)

x = np.arange(len(feature_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, human_means, width, label="Human", color="blue")
ax.bar(x, gpt3_means, width, label="GPT-3.5", color="orange")
ax.bar(x + width, gpt4_means, width, label="GPT-4.0", color="green")

ax.set_xlabel("Linguistic Features")
ax.set_ylabel("Mean Value (Scaled)")
ax.set_title("Comparison of Linguistic Features: Human vs GPT-3.5 vs GPT-4.0")
ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("linguistic_features_comparison.png")
plt.savefig("all 3 compared.png")
linguistic_features_df = pd.DataFrame(
    np.vstack([human_features, gpt3_features, gpt4_features]),
    columns=feature_names
)
linguistic_features_df["Source"] = (
    ["Human"] * len(human_features) +
    ["GPT-3.5"] * len(gpt3_features) +
    ["GPT-4.0"] * len(gpt4_features)
)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
axes = axes.flatten()
for i, feature in enumerate(feature_names):
    sns.boxplot(data=linguistic_features_df, x="Source", y=feature, ax=axes[i], palette="Set2")
    axes[i].set_title(f"Box Plot: {feature}")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Value")
plt.tight_layout()
plt.savefig("linguistic_features_boxplots.png")
plt.show()

print("Training classifier for feature importance...")
X_train, X_test, y_train, y_test = train_test_split(linguistic_features, labels, test_size=0.2, random_state=6960)

clf = RandomForestClassifier(n_estimators=100, random_state=23)
clf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)
feature_importance.to_csv("feature_importance.csv", index=False)
