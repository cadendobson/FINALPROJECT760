import random

files = ["NewsSA.txt", "sesotho_llm_headlines.txt"]
all_pairs = []

for filename in files:
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines)-1, 2):
            headline = lines[i].strip()
            label = lines[i+1].strip()
            all_pairs.append((headline, label))

random.shuffle(all_pairs)

with open("joined_shuffled_dataset.txt", "w", encoding="utf-8") as f:
    for headline, label in all_pairs:
        f.write(f"{headline}\n{label}\n")

print(f"Done! {len(all_pairs)} headline-label pairs written to joined_shuffled_dataset.txt")