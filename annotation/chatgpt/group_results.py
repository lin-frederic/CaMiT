import os
import json
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group batch results.")
    parser.add_argument("--batch_results_dir", type=str, help="Directory containing batch results", default="batch_results")
    parser.add_argument("--output_file", type=str, help="Path to output file", default="grouped_results.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.output_file):
        results = []
        for file in tqdm(os.listdir(args.batch_results_dir)):
            if file.endswith(".jsonl"):
                with open(os.path.join(args.batch_results_dir, file), "r") as f:
                    for line in f:
                        data = json.loads(line)
                        results.append(data)

        with open(args.output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    else:
        with open(args.output_file, "r") as f:
            results = [json.loads(line) for line in f]

    print(f"Total results: {len(results)}")
