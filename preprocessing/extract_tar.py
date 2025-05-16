import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse



def extract_tar(tar_file, dest_folder):
    try:
        with tarfile.open(tar_file, "r") as tar:
            tar.extractall(path=dest_folder)
    except Exception as e:
        print(f"Error extracting {tar_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tar files")
    parser.add_argument("--tar_folder", type=str, default=f"{os.environ['HOME']}/car_models_tars_missing")
    parser.add_argument("--dest_folder", type=str, default=f"{os.environ['HOME']}/cars_model_images")
    args = parser.parse_args()

    os.makedirs(args.dest_folder, exist_ok=True)

    tar_folder = [f for f in os.listdir(args.tar_folder) if f.endswith('.tar')]
    # filter out tar files that have already been extracted
    #tar_folder = [tar for tar in tar_folder if not os.path.exists(os.path.join(args.dest_folder, tar.split('.')[0]))]
    total_files = len(tar_folder)
    print(f"Extracting {total_files} tar files")

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_tar, os.path.join(args.tar_folder, tar), args.dest_folder) for tar in tar_folder]
        with tqdm(total=total_files) as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    print("All tar files extracted")


