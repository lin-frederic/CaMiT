import json
import pandas as pd
import matplotlib.pyplot as plt  

def normalize_class_name(s):
    return s.lower().replace("-", "_").replace(" ", "_")

def plot_aggregated_kid_vs_dynamics(kid_df: pd.DataFrame, method: str, scenario: str,
                                     class_dynamics_list: list, out_prefix: str):

    dynamics_df = pd.DataFrame(class_dynamics_list, columns=['filename', 'dynamics_score'])
    dynamics_df['class'] = dynamics_df['filename'].str.replace("results_", "", regex=False)\
                                               .str.replace(".json", "", regex=False)\
                                               .apply(normalize_class_name)
    
    kid_df['class'] = kid_df['class'].apply(normalize_class_name)
    filtered_kid_df = kid_df.query("method == @method and scenario == @scenario")

    merged_df = filtered_kid_df.merge(dynamics_df[['class', 'dynamics_score']], on='class', how='inner')
    if merged_df.empty:
        raise RuntimeError("No data after merging KID results with class dynamics")

    merged_df = merged_df.sort_values('dynamics_score')
    merged_df['dynamics_decile'] = pd.qcut(merged_df['dynamics_score'], 10, labels=False) + 1

    agg = (
        merged_df.groupby(['dynamics_decile', 'class'])  
                .agg(
                    mean_KID=('KID_value', 'mean'),
                    avg_dynamics_score=('dynamics_score', 'mean'),
                    total_samples_per_class_in_decile=('class', 'size'), 
                )
                .reset_index()
    )

    for decile in agg['dynamics_decile'].unique():
        decile_data = agg[agg['dynamics_decile'] == decile]
        print(f"Decile {decile}:")
        
        
        for _, row in decile_data.iterrows():
            class_name = row['class']
            total_samples = row['total_samples_per_class_in_decile']
            print(f"  Class: {class_name}, Total samples: {total_samples}")
        
        print("-" * 50)

    stats_csv = f"{out_prefix}_{method}_{scenario}_dynamics_decile_stats.csv"
    agg.to_csv(stats_csv, index=False)
    print(f"Saved decile statistics to {stats_csv}")


    plt.figure(figsize=(6, 4))
    plt.plot(agg['dynamics_decile'], agg['mean_KID'], marker='o', linestyle='-')
    plt.xlabel("Class Dynamics Decile (1 = most stable)")
    plt.ylabel("Mean KID (lower is better)")
    plt.title(f"{method} + {scenario}: Mean KID vs. Class Dynamics Decile")
    plt.grid(True)
    plt.tight_layout()

    plot_path = f"{out_prefix}_{method}_{scenario}_mean_kid_vs_dynamics_decile.png"
    plt.savefig(plot_path)
    print(f"Saved line plot to {plot_path}")
    plt.show()

def main():
    with open("label_scores.json", 'r') as f:
        class_dynamics = json.load(f)
    kid_df = pd.read_csv("individual_kid_values_clip_base.csv")

    plot_aggregated_kid_vs_dynamics(kid_df, "finetuned2", "with_year", class_dynamics, "yearsamples_finetuned2_with_year_decile_main")

if __name__ == "__main__":
    main()
