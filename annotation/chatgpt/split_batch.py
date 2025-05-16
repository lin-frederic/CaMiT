import os
import json
import math
import argparse

# Constants
# 200 MB - 1 MB
MAX_FILE_SIZE = 200000000 - 1000000

# Function to split JSONL file while ensuring each chunk is ≤ MAX_FILE_SIZE
def split_jsonl_file(file_path, max_size=MAX_FILE_SIZE):
    chunks = []
    current_chunk = []
    current_size = 0

    with open(file_path, 'r') as f:
        for line in f:
            line_size = len(line.encode('utf-8'))  # Get actual byte size
            if current_size + line_size > max_size:
                # Save current chunk
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += line_size

    # Append the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Create smaller batch files for each chunk
def create_batch_files(chunks, original_filename_path):
    batch_files = []
    os.makedirs(original_filename_path.replace(".jsonl", ""), exist_ok=True)
    
    for i, chunk in enumerate(chunks):
        batch_file = f"{original_filename_path.replace('.jsonl', '')}/batch_{i}.jsonl"
        with open(batch_file, "w") as f:
            f.writelines(chunk)
        batch_files.append(batch_file)
        print(f"Created batch file: {batch_file} ({os.path.getsize(batch_file)} bytes)")
    
    return batch_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split batch requests into smaller chunks.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file", default="chatgpt/batch_requests.jsonl")
    parser.add_argument("--max_size", type=int, help="Maximum file size in bytes", default=MAX_FILE_SIZE)
    args = parser.parse_args()

    chunks = split_jsonl_file(args.input_file, args.max_size)
    print(f"Split {args.input_file} into {len(chunks)} chunks.")
    batch_files = create_batch_files(chunks, args.input_file)
