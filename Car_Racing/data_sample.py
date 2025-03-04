# # import numpy as np
# # import matplotlib.pyplot as plt
# # import random
# # import os

# # # Define the correct data directory
# # data_dir = "data/test"

# # # List available .npz files
# # npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

# # # Load a random file
# # random_file = random.choice(npz_files)
# # data = np.load(os.path.join(data_dir, random_file))

# # # Extract stored images (original RGB format)
# # imgs = data["imgs"]  # Original RGB images

# # # Select random frames
# # num_images = min(5, len(imgs))  # Show up to 5 random images
# # random_indices = random.sample(range(len(imgs)), num_images)

# # # Plot random images
# # plt.figure(figsize=(15, 5))
# # for i, idx in enumerate(random_indices):
# #     plt.subplot(1, num_images, i + 1)
# #     plt.imshow(imgs[idx])
# #     plt.axis("off")
# #     plt.title(f"Frame {idx}")
# # plt.suptitle(f"Random Images from {random_file}")
# # plt.show()

# import numpy as np
# import os

# # Define the correct absolute path
# data_dir = "/Users/nrohithreddy/Desktop/TEA/Car_Racing/data/test"

# # Check if the directory exists
# if not os.path.exists(data_dir):
#     print(f"Error: Directory '{data_dir}' not found. Please check the path.")
# else:
#     # List available .npz files
#     npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

#     if not npz_files:
#         print("No .npz files found in the directory.")
#     else:
#         # Dictionary to store keys from all .npz files
#         key_summary = {}

#         # Iterate through each .npz file and extract keys
#         for file in npz_files:
#             file_path = os.path.join(data_dir, file)
#             data = np.load(file_path)

#             # Store keys from this file
#             key_summary[file] = list(data.keys())

#         # Display keys from each .npz file
#         for file, keys in key_summary.items():
#             print(f"File: {file}, Keys: {keys}")

#         # Display shapes and types for each key
#         for npz_file in npz_files:
#             file_path = os.path.join(data_dir, npz_file)
#             data = np.load(file_path)

#             print(f"\nFile: {npz_file}")
#             for key in data.keys():
#                 value = data[key]
#                 print(f"  Key: {key} | Type: {type(value)} | Shape: {value.shape} | Dtype: {value.dtype}")


# import numpy as np
# import os

# # Define the correct absolute path
# data_dir = "/Users/nrohithreddy/Desktop/TEA/Car_Racing/data/test"

# # List available .npz files
# npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

# # Initialize total counters
# total_safe = 0
# total_unsafe = 0

# # Dictionary to store counts of safe values per file
# safe_counts = {}

# # Iterate through each .npz file and count safe values
# for npz_file in npz_files:
#     file_path = os.path.join(data_dir, npz_file)
#     data = np.load(file_path)

#     if "safe" in data:
#         safe_array = data["safe"]
#         count_safe = np.sum(safe_array == 1)  # Count occurrences of 1 (safe)
#         count_unsafe = np.sum(safe_array == 0)  # Count occurrences of 0 (unsafe)

#         # Update total counts
#         total_safe += count_safe
#         total_unsafe += count_unsafe

#         # Store count for the current file
#         safe_counts[npz_file] = {"Safe (1)": count_safe, "Unsafe (0)": count_unsafe}

# # Display the safe and unsafe counts for each file
# print("\nSafe Values Count per File:")
# for file, counts in safe_counts.items():
#     print(f"File: {file} | Safe (1): {counts['Safe (1)']} | Unsafe (0): {counts['Unsafe (0)']}")

# # Display total counts
# print("\nTotal Count Across All Files:")
# print(f"Total Safe (1) Count: {total_safe}")
# print(f"Total Unsafe (0) Count: {total_unsafe}")


import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Define the correct data directory
data_dir = "data/test"

# List all .npz files
npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

# Ensure files exist
if not npz_files:
    print("No .npz files found in the directory.")
    exit()

# Load all images marked as 'unsafe' (safe == 0)
unsafe_images = []

for file in npz_files:
    file_path = os.path.join(data_dir, file)
    data = np.load(file_path)

    # Extract images and safety labels
    imgs = data["imgs"]  # Original RGB images
    safe_flags = data["safe"]  # Safety flags (0 = off-road, 1 = on-road)

    # Get only unsafe images
    for i in range(len(safe_flags)):
        if safe_flags[i] == 0:
            unsafe_images.append(imgs[i])

# Check if we have unsafe images
if not unsafe_images:
    print("No unsafe images found in the dataset.")
    exit()

# Randomly select up to 5 unsafe images to display
num_images = min(5, len(unsafe_images))
random_indices = random.sample(range(len(unsafe_images)), num_images)

# Plot images
plt.figure(figsize=(15, 5))
for i, idx in enumerate(random_indices):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(unsafe_images[idx])
    plt.axis("off")
    plt.title(f"Unsafe Image {i+1}")
plt.suptitle("Random Unsafe (Off-Road) Images from Dataset")
plt.show()
