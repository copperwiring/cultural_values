from datasets import load_dataset
import pandas as pd
from PIL import Image
import os
import io
from tqdm import tqdm

# Load the dataset
dataset = load_dataset('afaji/cvqa')

# Define the path to save the images and CSV file
base_dir = "data/cvqa/"
image_save_path = os.path.join(base_dir, "images/")
csv_save_path = os.path.join(base_dir, "metadata.csv")

# Create directories if they do not exist
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

# List to store the rows for the CSV file
csv_rows = []

# Iterate through the test dataset
for i, row in enumerate(tqdm(dataset['test'])):
    image_data = row['image']
    image_id = row['ID']  # Assuming 'ID' uniquely identifies the image

    # Save the image
    image_file_name = f"{image_id}.png"  # Save image with its ID as filename
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        image = image_data
    
    # Save the image to the specified directory
    image.save(f"{image_save_path}{image_file_name}")

    # Collect the metadata for the CSV row
    csv_row = {
        "ID": row["ID"],
        "Subset": row["Subset"],
        "Question": row["Question"],
        "Translated Question": row["Translated Question"],
        "Options": row["Options"],
        "Translated Options": row["Translated Options"],
        "Label": row["Label"],
        "Category": row["Category"],
        "Image Type": row["Image Type"],
        "Image Source": row["Image Source"],
        "License": row["License"],
        "Image File Name": image_file_name  # Link image file name to CSV
    }
    csv_rows.append(csv_row)

# Convert the list of rows into a pandas DataFrame
df = pd.DataFrame(csv_rows)

# Save the DataFrame to a CSV file in the specified directory
df.to_csv(csv_save_path, index=False)

print(f"Images saved in {image_save_path}")
print(f"CSV metadata saved at {csv_save_path}")
