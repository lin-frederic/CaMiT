
"""
Decile analysis of KID vs. training-set size for SD 1.5 LoRA fine-tuned models.

Usage:
    python decile_kid_analysis.py \
        --train_json path/to/train_flickr.json \
        --kid_csv path/to/sdxl_kid_results.csv \
        --method finetuned2 \
        --scenario with_year \
        --out_prefix figure7
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def count_training_samples(train_json_path: str) -> pd.DataFrame:
    
    with open(train_json_path, 'r') as f:
        data = json.load(f)

    records = []
    for entry in data.values():
        year = entry.get("time") or entry.get("metadata", {}).get("date")
        try:
            year = int(year)
        except Exception:
            continue

       
        classes = {
            box.get("class")
            for box in entry.get("boxes", [])
            if box.get("class") and box["class"] != "unknown"
        }
        for cls in classes:
            records.append((cls, year))

    df = pd.DataFrame(records, columns=['class', 'year'])
    counts = (
        df.groupby(['class', 'year'])
          .size()
          .reset_index(name='n_train_samples')
    )
    return counts


def load_kid_csv(kid_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(kid_csv_path)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    if 'KID_value' in df.columns:
        df = df.rename(columns={'KID_value': 'KID'})

    return df


def decile_kid(
    counts_df: pd.DataFrame,
    kid_df: pd.DataFrame,
    method: str,
    scenario: str,
    out_prefix: str
) -> None:
    
    df = kid_df.query("method == @method and scenario == @scenario")
    if df.empty:
        raise RuntimeError(f"No rows found for method={method}, scenario={scenario}")

    df = df.merge(counts_df, on=['class', 'year'], how='inner')
    if df.empty:
        raise RuntimeError("Merge of KID data and sample counts yielded no rows")
    df['decile'] = pd.qcut(df['n_train_samples'], 10, labels=False) + 1

    agg = (
        df.groupby('decile')
          .agg(
              mean_KID     = ('KID', 'mean'),
              avg_samples  = ('n_train_samples', 'mean'),
              slice_count  = ('class', 'size'),
          )
          .reset_index()
    )

    stats_csv = f"{out_prefix}_{method}_{scenario}_decile_stats.csv"
    agg.to_csv(stats_csv, index=False)
    
    print(f"Saved decile statistics to {stats_csv}")
    plt.figure(figsize=(6, 4))
    plt.plot(agg.decile, agg.mean_KID, marker='o', linestyle='-')
    plt.xlabel("Training-set size decile (1 = smallest)")
    plt.ylabel("Mean KID (lower is better)")
    plt.title(f"{method} + {scenario}: Mean KID vs. Training-Size Decile")
    plt.grid(True)
    plt.tight_layout()

    plot_png = f"{out_prefix}_{method}_{scenario}_decile_plot.png"
    plt.savefig(plot_png)
    print(f"Saved decile plot to {plot_png}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot mean KID vs. training-set size deciles"
    )
    parser.add_argument(
        "--train_json", required=True,
        help="Path to the Flickr-style train JSON"
    )
    parser.add_argument(
        "--kid_csv", required=True,
        help="CSV containing [model_name,method,scenario,class,year,KID_value]"
    )
    parser.add_argument(
        "--method", required=True,
        choices=["finetuned1", "finetuned2", "plain_sd"],
        help="Generation method to analyze"
    )
    parser.add_argument(
        "--scenario", required=True,
        choices=["with_year", "without_year"],
        help="Scenario to analyze"
    )
    parser.add_argument(
        "--out_prefix", default="figure7",
        help="Prefix for output files (CSV & PNG)"
    )

    args = parser.parse_args()

    counts_df = count_training_samples(args.train_json)
    kid_df = load_kid_csv(args.kid_csv)
    decile_kid(
        counts_df,
        kid_df,
        method    = args.method,
        scenario  = args.scenario,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
