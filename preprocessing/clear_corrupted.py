"""
This script removes corrupted images from a dataset. It checks each image in the specified directory and deletes it if its size is 0 bytes.
"""
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

def process_image(image_path):
    """Remove the image if its size is 0 bytes."""
    if os.path.getsize(image_path) == 0:
        os.remove(image_path)

def process_class_images(class_path, max_workers=8):
    """Process all images in a single class directory."""
    images = os.listdir(class_path)
    image_paths = [os.path.join(class_path, image) for image in images]

    # Parallelize image processing within the class
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_image, image_paths), 
                  total=len(image_paths), 
                  desc=f"Processing images in {os.path.basename(class_path)}"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to data", default=os.path.join(os.environ['HOME'], 'cars_images'))
    parser.add_argument("--workers_per_class", type=int, help="Number of threads per class", default=16)
    args = parser.parse_args()

    data_path = args.data_path
    classes = os.listdir(data_path)

    # Parallelize across classes
    with ThreadPoolExecutor() as class_executor:
        futures = [
            class_executor.submit(process_class_images, os.path.join(data_path, class_name), args.workers_per_class)
            for class_name in classes
        ]

        # Monitor progress of class-level processing
        for future in tqdm(futures, desc="Processing classes"):
            future.result()  # Wait for completion
