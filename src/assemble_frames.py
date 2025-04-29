import os
import re
import matplotlib.pyplot as plt
from PIL import Image

# CONFIGURE THIS
input_folder = "unzipped_projection_frames"  # <-- folder where you extracted all your .png files
output_folder = "assembled_sequences"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Mapping from file keywords to graph type
graph_types = {
    "worm": "worm",
    "random": "random",
    "social": "social",
    "knowledge": "knowledge",
    "protein": "protein"
}

# Helper to extract noise level from filename
def extract_noise_level(filename):
    match = re.search(r"(\d+)%", filename)
    if match:
        return int(match.group(1))
    else:
        # Check for filenames without noise level (e.g., "worm_brain_initial.png")
        if "initial" in filename.lower() or "%" not in filename:
             return -1 # Assign -1 to ensure initial frames come first
        # If still no match, perhaps default to 0 or raise an error?
        print(f"Warning: Could not extract noise level from {filename}. Defaulting to 0.")
        return 0


# Helper to determine graph type from filename
def determine_graph_type(filename):
    filename_lower = filename.lower()
    # Handle potential "worm_brain" case specifically if needed
    if "worm_brain" in filename_lower or "worm brain" in filename_lower:
        return "worm"
    for key in graph_types:
        if key in filename_lower:
            return graph_types[key]
    print(f"Warning: Could not determine graph type for {filename}.")
    return None  # if no known graph type

# Gather all images
all_images = {}

print(f"Scanning for images in: {os.path.abspath(input_folder)}")
if not os.path.isdir(input_folder):
    print(f"Error: Input folder '{input_folder}' not found. Please create it and place your images inside.")
    exit()

for fname in os.listdir(input_folder):
    if fname.lower().endswith(".png"):
        gtype = determine_graph_type(fname)
        if gtype:
            if gtype not in all_images:
                all_images[gtype] = []
            # Store tuple (noise_level, filename) for easier sorting
            noise_level = extract_noise_level(fname)
            all_images[gtype].append((noise_level, fname))
        else:
             print(f"Skipping file with undetermined type: {fname}")


# Assemble each graph type
if not all_images:
    print("No images found or processed. Exiting.")
    exit()

for gtype, files_with_noise in all_images.items():
    print(f"Assembling frames for: {gtype}")

    # Sort by noise level (using the stored tuple)
    files_sorted_tuples = sorted(files_with_noise, key=lambda x: x[0])
    # Extract just the filenames in sorted order
    files_sorted = [f[1] for f in files_sorted_tuples]

    if not files_sorted:
        print(f"No images found for type {gtype} after sorting. Skipping.")
        continue

    try:
        images = [Image.open(os.path.join(input_folder, f)) for f in files_sorted]
    except Exception as e:
        print(f"Error opening images for {gtype}: {e}. Skipping this type.")
        continue

    # Ensure all images were loaded
    if not images:
         print(f"Failed to load any images for {gtype}. Skipping.")
         continue


    # Filter out potentially problematic images (e.g., zero height/width)
    valid_images = [img for img in images if img.height > 0 and img.width > 0]
    if not valid_images:
        print(f"No valid images (with height > 0) found for {gtype}. Skipping.")
        continue

    # Resize all images to the same height (using minimum height of *valid* images)
    try:
        min_height = min(img.height for img in valid_images)
        images_resized = []
        for img in valid_images:
             aspect_ratio = img.width / img.height
             new_width = int(aspect_ratio * min_height)
             images_resized.append(img.resize((new_width, min_height), Image.Resampling.LANCZOS)) # Use LANCZOS for better quality
    except ValueError:
         print(f"Could not determine minimum height for {gtype}, possibly due to invalid image dimensions. Skipping.")
         continue
    except Exception as e:
         print(f"Error during resizing for {gtype}: {e}. Skipping.")
         continue


    if not images_resized:
        print(f"No images available for stitching after resizing for {gtype}. Skipping.")
        continue

    # Stitch horizontally
    total_width = sum(img.width for img in images_resized)
    try:
        stitched = Image.new('RGB', (total_width, min_height)) # Use min_height determined earlier
    except Exception as e:
        print(f"Error creating new image canvas for {gtype} (size {total_width}x{min_height}): {e}. Skipping.")
        continue


    x_offset = 0
    for img in images_resized:
        stitched.paste(img, (x_offset, 0))
        x_offset += img.width
        # Close images after use
        img.close()

    # Save final composite
    output_path = os.path.join(output_folder, f"melting_sequence_{gtype}.png")
    try:
        stitched.save(output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error saving stitched image {output_path}: {e}")
    finally:
        # Ensure the stitched image is closed
        stitched.close()


print("\\n✅ All melting sequences assembled!") 