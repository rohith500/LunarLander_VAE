import numpy as np
import os

# Define the correct absolute path
data_dir = "/Users/nrohithreddy/Desktop/TEA/Car_Racing/data/test"

# List available .npz files
npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

# Initialize total counters
total_safe = 0
total_unsafe = 0

# Dictionary to store counts of safe values per file
safe_counts = {}

# Iterate through each .npz file and count safe values
for npz_file in npz_files:
    file_path = os.path.join(data_dir, npz_file)
    data = np.load(file_path)

    if "safe" in data:
        safe_array = data["safe"]
        count_safe = np.sum(safe_array == 1)  # Count occurrences of 1 (safe)
        count_unsafe = np.sum(safe_array == 0)  # Count occurrences of 0 (unsafe)

        # Update total counts
        total_safe += count_safe
        total_unsafe += count_unsafe

        # Store count for the current file
        safe_counts[npz_file] = {"Safe (1)": count_safe, "Unsafe (0)": count_unsafe}

# Display the safe and unsafe counts for each file
print("\nSafe Values Count per File:")
for file, counts in safe_counts.items():
    print(f"File: {file} | Safe (1): {counts['Safe (1)']} | Unsafe (0): {counts['Unsafe (0)']}")

# Display total counts
print("\nTotal Count Across All Files:")
print(f"Total Safe (1) Count: {total_safe}")
print(f"Total Unsafe (0) Count: {total_unsafe}")
