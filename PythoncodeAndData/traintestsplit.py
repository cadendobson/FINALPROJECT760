import random

inputFile = "joined_shuffled_dataset.txt"
trainFile = "train.txt"
testFile = "test.txt"
testRatio = 0.2

with open(inputFile, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
pairs = [(lines[i], lines[i+1]) for i in range(0, len(lines)-1, 2)]

random.shuffle(pairs)
splitIdx = int(len(pairs) * (1 - testRatio))
trainPairs = pairs[:splitIdx]
testPairs = pairs[splitIdx:]

with open(trainFile, "w", encoding="utf-8") as f:
    for headline, label in trainPairs:
        f.write(f"{headline}\n{label}\n")

with open(testFile, "w", encoding="utf-8") as f:
    for headline, label in testPairs:
        f.write(f"{headline}\n{label}\n")

print(f"Done! {len(trainPairs)} training pairs and {len(testPairs)} test pairs written.")