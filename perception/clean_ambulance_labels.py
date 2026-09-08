from pathlib import Path

# Clean dataset location
DATASET = Path("data/ambulance_clean")

splits = ["train", "valid", "test"]

for split in splits:
    labels_dir = DATASET / split / "labels"

    print(f"\nProcessing: {split}")

    for label_file in labels_dir.glob("*.txt"):

        new_lines = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(parts[0])

                # Ambulance / ambulance → class 0
                if class_id in [0, 1]:
                    parts[0] = "0"
                    new_lines.append(" ".join(parts))

                # Siren → ignore
                elif class_id == 2:
                    continue

        with open(label_file, "w") as f:
            f.write("\n".join(new_lines))

    print(f"Finished: {split}")

print("\nCleaning completed successfully!")