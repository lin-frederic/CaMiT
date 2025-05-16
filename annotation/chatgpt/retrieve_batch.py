import os
from openai import OpenAI
import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve batch job results.")
    parser.add_argument("--api_key", type=str, help="OpenAI API key", required=True)
    args = parser.parse_args()
    os.makedirs("batch_errors", exist_ok=True)
    os.makedirs("batch_results", exist_ok=True)
    client = OpenAI(api_key=args.api_key)

    with open(args.batch_ids_file, "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        input_file, batch_id = line.split(",")
        input_file = input_file.split("/")[-1].split(".")[0]
        if os.path.exists(f"batch_results/{input_file}_results.jsonl"):
            print(f"Results file already exists for {input_file}. Skipping.")
            continue
        batch_job = client.batches.retrieve(batch_id)
        print(f"Batch job for {input_file}:")   
        print(batch_job)
        result_file_id = batch_job.output_file_id
        if result_file_id is not None:
            result = client.files.content(result_file_id)
            with open(f"batch_results/{input_file}_results.jsonl", "wb") as f:
                f.write(result.read())
            print(f"Results file: batch_results/{input_file}_results.jsonl.")

        error_file_id = batch_job.error_file_id
        if error_file_id is not None:
            error_file = client.files.retrieve(error_file_id)
            content = client.files.content(error_file_id)
            with open(f"batch_errors/{input_file}_error.jsonl", "wb") as f:
                f.write(content.read())
            print(f"Error file: batch_errors/{input_file}_error.jsonl.")

        