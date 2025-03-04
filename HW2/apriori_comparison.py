import pandas as pd
import time
import argparse
import os
import sys

# Function to run an Apriori implementation and return the counts
def run_apriori(apriori_func, df, colname, min_support):
    start_time = time.time()
    result = apriori_func(df, colname, min_support)
    end_time = time.time()
    
    num_itemsets = len(result[0])
    
    return {
        'comparison_count': result[1],
        'subset_check_count': result[2],
        'trie_insert_count': result[3] if len(result) > 3 else 0,
        'num_itemsets': num_itemsets,
        'running_time': end_time - start_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Apriori implementations")
    parser.add_argument("--input", type=str, default="data.csv", help="Input file path")
    parser.add_argument("--min_support", type=int, default=500, help="Minimum support count")
    parser.add_argument("--colname", type=str, default="text_keywords", help="Column name for transactions")
    args = parser.parse_args()
    
    # Load data
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_path = os.path.join(script_path, args.input)
    df = pd.read_csv(data_path)
    
    # Import Apriori implementations
    from apriori_k1km1 import apriori as apriori_k1km1
    from apriori_km1km1 import apriori as apriori_km1km1
    from apriori_trie_improved import apriori_trie as apriori_trie_improved
    
    # Run and compare the implementations
    results = {}
    results['Apriori_k1km1'] = run_apriori(apriori_k1km1, df, args.colname, args.min_support)
    results['Apriori_km1km1'] = run_apriori(apriori_km1km1, df, args.colname, args.min_support)
    results['Apriori_trie_improved'] = run_apriori(apriori_trie_improved, df, args.colname, args.min_support)
    
    # Print results
    print(f"{'='*15}< Results >{'='*15}")
    for name, result in results.items():
        print(f"Algorithm: {name}")
        print(f"  Running Time: {result['running_time']:.4f} seconds")
        print(f"  Comparison Count: {result['comparison_count']}")
        print(f"  Subset Check Count: {result['subset_check_count']}")
        print(f"  Trie Insert Count: {result['trie_insert_count']}")
        print(f"  Number of Frequent Itemsets: {result['num_itemsets']}")
        print("-" * 40)
