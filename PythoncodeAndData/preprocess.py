import random
import re

def clean_headline(text):

    text = text.lower()
    text = re.sub(r"[^a-zà-ÿ'’\-\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

files = ["NewsSA.txt", "sesotho_llm_headlines.txt"]
all_pairs = []
seen = set()

for filename in files:
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines)-1, 2):
            headline = clean_headline(lines[i])
            label = lines[i+1].strip()
            label = "1" if label == "1" else "0"
            if not headline or not label:
                continue
            pair = (headline, label)
            if pair not in seen:
                all_pairs.append(pair)
                seen.add(pair)

random.shuffle(all_pairs)

with open("joined_shuffled_dataset.txt", "w", encoding="utf-8") as f:
    for headline, label in all_pairs:
        f.write(f"{headline}\n{label}\n")

print(f"Done! {len(all_pairs)} cleaned and shuffled headline-label pairs written to joined_shuffled_dataset.txt")