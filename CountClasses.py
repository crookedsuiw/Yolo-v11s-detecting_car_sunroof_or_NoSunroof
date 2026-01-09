dir = ''

import os
from collections import Counter

labels_dir = "train/labels"
classes_file = 'classes.txt'
with open(classes_file, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]
class_counts = Counter()

for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue

    file_path = os.path.join(labels_dir, file)
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

# Print results
for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_names[class_id]}: {count}")
