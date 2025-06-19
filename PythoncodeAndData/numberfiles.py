with open("NewsSA.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("NewsSA.txt", "w", encoding="utf-8") as f:
    for line in lines:
        if line.strip() in {"-1", "0", "1"}:
            f.write("0\n")
        else:
            f.write(line)