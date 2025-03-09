import os
import glob

data_folder = "data/labels/"

def process_label_files(folder):
    label_ranges = {}

    label_files = glob.glob(os.path.join(folder, "*.txt"))

    for file_path in label_files:
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue  # Skip malformed lines

                label_name = parts[0]
                width = float(parts[4])
                length = float(parts[5])
                height = float(parts[6])

                if label_name not in label_ranges:
                    label_ranges[label_name] = {
                        "width": {"min": width, "max": width},
                        "length": {"min": length, "max": length},
                        "height": {"min": height, "max": height},
                    }
                else:
                    label_ranges[label_name]["width"]["min"] = min(label_ranges[label_name]["width"]["min"], width)
                    label_ranges[label_name]["width"]["max"] = max(label_ranges[label_name]["width"]["max"], width)
                    
                    label_ranges[label_name]["length"]["min"] = min(label_ranges[label_name]["length"]["min"], length)
                    label_ranges[label_name]["length"]["max"] = max(label_ranges[label_name]["length"]["max"], length)
                    
                    label_ranges[label_name]["height"]["min"] = min(label_ranges[label_name]["height"]["min"], height)
                    label_ranges[label_name]["height"]["max"] = max(label_ranges[label_name]["height"]["max"], height)

    return label_ranges

config = process_label_files(data_folder)

# Print results
for label, ranges in config.items():
    print(f"{label}:")
    print(f"  Width  - Min: {ranges['width']['min']}, Max: {ranges['width']['max']}")
    print(f"  Length - Min: {ranges['length']['min']}, Max: {ranges['length']['max']}")
    print(f"  Height - Min: {ranges['height']['min']}, Max: {ranges['height']['max']}")
