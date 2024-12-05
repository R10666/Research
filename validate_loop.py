import os
import re
import torch
import csv

# Define paths
weights_folder = "./weights"  # Path to your folder containing weight files
csv_file = "validation_metrics.csv"

# Regex to extract epoch number from file names like "model_epoch_10.pt"
def extract_epoch(filename):
    match = re.search(r"epoch_(\d+)\.pt", filename)
    return int(match.group(1)) if match else None

# Sort weight files by epoch number
weight_files = [
    f for f in os.listdir(weights_folder) if f.endswith(".pt")
]
sorted_files = sorted(weight_files, key=extract_epoch)

# CSV Header
header = ["epoch", "BLEU", "WER", "CER"]

# Ensure CSV file exists and write the header
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loop through weight files
for weight_file in sorted_files:
    epoch = extract_epoch(weight_file)
    if epoch is None:
        continue

    print(f"Validating model from epoch {epoch}...")
    
    # Load weights
    weight_path = os.path.join(weights_folder, weight_file)
    state = torch.load(weight_path)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    # Run validation
    with torch.no_grad():
        cer, wer, bleu = run_validation(model, validation_ds, device)  # Your validation logic here

    # Log metrics
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, bleu, wer, cer])

    print(f"Metrics for epoch {epoch} logged.")

print("Validation completed for all weights.")
