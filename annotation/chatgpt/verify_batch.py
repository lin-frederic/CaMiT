from openai import OpenAI
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify batch job status.")
    parser.add_argument("--api_key", type=str, help="OpenAI API key", required=True)
    parser.add_argument("--batch_ids_file", type=str, help="Path to file containing batch IDs", default="batch_ids.txt")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)
    with open(args.batch_ids_file, "r") as f:
        lines = f.read().splitlines()
    
    completed_jobs = []
    for line in lines:
        input_file, batch_id = line.split(",")
        batch_job = client.batches.retrieve(batch_id)
        print(f"Batch job for {input_file}:")   
        print(batch_job)
        if batch_job.status == "completed":
            completed_jobs.append(input_file)

    print(f"Completed jobs: {completed_jobs}")
    print(f"Total completed jobs: {len(completed_jobs)}")
    print(f"Total jobs: {len(lines)}")
    