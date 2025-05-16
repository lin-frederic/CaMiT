import os
import json
from openai import OpenAI
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload batch requests to OpenAI.")
    parser.add_argument("--api_key", type=str, help="OpenAI API key", required=True)
    # add list of jsonl files
    parser.add_argument("--batch_requests_folder", type=str, help="Folder containing batch requests", default="batch_requests")
    parser.add_argument("--batch_ids", type=str, help="File containing batch IDs", default="batch_ids.txt")
    # add a limit to the number of files to upload
    parser.add_argument("--limit", type=int, help="Limit the number of files to upload", default=None)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)
    batch_requests = os.listdir(args.batch_requests_folder)
    batch_requests = sorted(batch_requests, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    batch_requests = [f"{args.batch_requests_folder}/{batch_request}" for batch_request in batch_requests if batch_request.endswith(".jsonl")]

    if args.limit:
        batch_requests = batch_requests[:args.limit]
        
    if os.path.exists(args.batch_ids):
        with open(args.batch_ids, "r") as f:
            batch_ids = f.readlines()
        # only process the files that are not already uploaded
        processed_files = [batch_id.split(",")[0] for batch_id in batch_ids]
        print(f"Already processed files: {processed_files}")
        batch_requests = [batch_request for batch_request in batch_requests if batch_request not in processed_files]

    print(f"Uploading {batch_requests}")
    for request_file in batch_requests:
        batch_file = client.files.create(
            file=open(request_file, "rb"),
            purpose='batch')
        print(f"Uploaded batch file: {batch_file.id} ({request_file})")

        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"Created batch job: {batch_job.id} ({request_file})")

        with open(args.batch_ids, "a") as f:
            f.write(f"{request_file},{batch_job.id}\n")


# python chatgpt/upload_batch.py --requests test_chatgpt/batch_requests_chunk_0.jsonl test_chatgpt/batch_requests_chunk_1.jsonl test_chatgpt/batch_requests_chunk_2.jsonl

    
