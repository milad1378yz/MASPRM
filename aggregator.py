import os
import re
import csv
import statistics
from collections import defaultdict


LOG_DIR = "logs"
OUTPUT_CSV = "aggregated_results.csv"


def get_experiment_id_and_seed(filename):
    """
    Splits filename to separate the experiment configuration from the seed.
    Expects format: {Experiment_Config}_{Seed}.txt
    Returns: (experiment_config, seed)
    """
    name_stem = filename.replace(".txt", "")

    # Find the last underscore to separate seed
    if "_" in name_stem:
        exp_id, seed = name_stem.rsplit("_", 1)
        if seed.isdigit():
            return exp_id, seed

    # Fallback if no seed found
    return name_stem, "unknown"


def parse_log_content(file_path):
    """
    Reads a log file and extracts metrics based on the specific
    requirements: Pass@1, Pass@3, Pass@5, and other usage metrics.
    """
    metrics = {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Parse Accuracy (Pass@1)
        # Pattern: Accuracy (pass@1): 0.1234
        acc_match = re.search(r"Accuracy \(pass@1\):\s*([\d\.]+)", content)
        if acc_match:
            metrics["pass@1"] = float(acc_match.group(1))

        # 2. Parse variable Pass@k
        # Pattern: Pass@2: 0.1234
        # We find all occurrences, but filter ONLY for pass@3 and pass@5 per instructions
        pass_k_matches = re.findall(r"Pass@(\d+):\s*([\d\.]+)", content)
        for k, val in pass_k_matches:
            if k in ['3', '5']:
                metrics[f"pass@{k}"] = float(val)

        # 3. Parse Tokens
        # Pattern: Tokens: 123.4 ± 10.1 (50 examples)
        tok_match = re.search(r"Tokens:\s*([\d\.]+)\s*±", content)
        if tok_match:
            metrics["tokens"] = float(tok_match.group(1))

        # 4. Parse PRM Calls (Mean)
        # Pattern: Mean PRM calls per example: 12.55
        prm_match = re.search(r"Mean PRM calls per example:\s*([\d\.]+)", content)
        if prm_match:
            metrics["prm_calls"] = float(prm_match.group(1))

        # 5. Parse Agent Runs (Mean)
        # Pattern: Mean Agent runs per example: 1.05
        agent_match = re.search(r"Mean Agent runs per example:\s*([\d\.]+)", content)
        if agent_match:
            metrics["agent_runs"] = float(agent_match.group(1))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    return metrics


# ==========================================
# 3. MAIN AGGREGATION LOGIC
# ==========================================


def main():
    # Store data as: data[experiment_id][metric_name] = [val_seed1, val_seed2, ...]
    aggregated_data = defaultdict(lambda: defaultdict(list))

    # 1. Read Files
    if not os.path.exists(LOG_DIR):
        print(f"Directory '{LOG_DIR}' not found.")
        return

    files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith(".txt")])
    print(f"Found {len(files)} log files. Parsing...")

    for filename in files:
        exp_id, seed = get_experiment_id_and_seed(filename)
        file_path = os.path.join(LOG_DIR, filename)

        metrics = parse_log_content(file_path)

        if metrics:
            for key, value in metrics.items():
                aggregated_data[exp_id][key].append(value)

    # 2. Calculate Statistics
    final_rows = []

    # Determine all unique metrics found across all files
    all_metrics = set()
    for exp_data in aggregated_data.values():
        all_metrics.update(exp_data.keys())

    # Define metrics that should NOT have a standard deviation column
    no_std_metrics = {"pass@3", "pass@5"}

    # Sort metrics for consistent column ordering
    def metric_sort_key(m):
        if m == "pass@1":
            return "0"
        if m.startswith("pass@"):
            return f"1_{m}"
        return f"2_{m}"

    sorted_metrics = sorted(list(all_metrics), key=metric_sort_key)

    for exp_id, metrics_map in aggregated_data.items():
        row = {"Experiment": exp_id}

        # Calculate how many seeds were found for this experiment
        n_seeds = max((len(v) for v in metrics_map.values()), default=0)
        row["Seeds"] = n_seeds

        for metric in sorted_metrics:
            values = metrics_map.get(metric, [])

            if not values:
                row[f"{metric}_mean"] = "N/A"
                if metric not in no_std_metrics:
                    row[f"{metric}_std"] = "N/A"
                continue

            # Calculate Mean
            mean_val = statistics.mean(values)
            row[f"{metric}_mean"] = f"{mean_val:.4f}"

            # Calculate Std Dev ONLY if the metric is not in the exclusion list
            if metric not in no_std_metrics:
                if len(values) > 1:
                    std_val = statistics.stdev(values)
                    row[f"{metric}_std"] = f"{std_val:.4f}"
                else:
                    row[f"{metric}_std"] = "0.0000"

        final_rows.append(row)

    # 3. Write to CSV
    if not final_rows:
        print("No valid data found.")
        return

    # Dynamic Fieldnames construction
    fieldnames = ["Experiment", "Seeds"]
    for metric in sorted_metrics:
        fieldnames.append(f"{metric}_mean")
        # Only add _std column if not in the exclusion list
        if metric not in no_std_metrics:
            fieldnames.append(f"{metric}_std")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)

    print("-" * 60)
    print(f"Aggregation complete.")
    print(f"Processed {len(aggregated_data)} unique experiments.")
    print(f"Results saved to: {os.path.abspath(OUTPUT_CSV)}")
    print("-" * 60)


if __name__ == "__main__":
    main()