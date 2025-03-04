import time
import pandas as pd
import os
import sys
import argparse

# Import all three Apriori implementations
from apriori_k1km1 import apriori as apriori_k1km1
from apriori_km1km1 import apriori as apriori_km1km1
from apriori_trie_improved import apriori_trie as apriori_trie_improved

def run_experiment(df, colname, start, end, step):
    records = []
    
    # Dictionary mapping algorithm names to their functions.
    algorithms = {
        'Apriori_k1km1': apriori_k1km1,
        'Apriori_km1km1': apriori_km1km1,
        'Apriori_trie_improved': apriori_trie_improved
    }

    for thresh in range(start, end + 1, step):
        for alg_name, alg_func in algorithms.items():
            t0 = time.time()
            result = alg_func(df, colname, thresh)
            t1 = time.time()
            running_time = t1 - t0
            num_itemsets = len(result[0])
            comparison_count = result[1]
            subset_check_count = result[2]
            trie_insert_count = result[3] if len(result) > 3 else 0

            cur = {
                'algorithm': alg_name,
                'min_support': thresh,
                'running_time': running_time,
                'num_itemsets': num_itemsets,
                'comparison_count': comparison_count,
                'subset_check_count': subset_check_count,
                'trie_insert_count': trie_insert_count
            }
            records.append(cur)

            for key, value in cur.items():
                print(f"{key}: {value}")
            print("-" * 40)

    return pd.DataFrame(records)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment: min_support vs runtime for Apriori algorithms")
    parser.add_argument("--input", type=str, default="data.csv", help="Input file path")
    parser.add_argument("--colname", type=str, default="text_keywords", help="Mining column name")
    parser.add_argument("--start", type=int, default=0, help="Starting min_support value")
    parser.add_argument("--end", type=int, default=1000, help="Ending min_support value")
    parser.add_argument("--step", type=int, default=50, help="Increment step for min_support")
    parser.add_argument("--output", type=str, default="min_support_experiment.csv", help="Output CSV file name")
    args = parser.parse_args()

    # Load dataset
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_path = os.path.join(script_path, args.input)
    df = pd.read_csv(data_path)

    # Run experiment and gather the statistics
    result_df = run_experiment(df, args.colname, args.start, args.end, args.step)
    output_path = os.path.join(script_path, args.output)
    result_df.to_csv(output_path, index=False)
    print("Experiment completed. Results saved to", output_path)
