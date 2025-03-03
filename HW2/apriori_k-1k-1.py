import argparse
import pandas as pd
import os
import sys


def parse_entry(entry) -> list:
    return entry.split(";")


def apriori(df, column_name, min_support):
    # Drop entries with empty item sets
    entries = df[column_name].dropna()

    # Immutable sets for hashing
    parsed_entries = [frozenset(parse_entry(entry)) for entry in entries]

    # Get item counts
    counts = {}
    for entry in parsed_entries:
        for item in entry:
            counts[item] = counts.get(item, 0) + 1
    
    # F_1
    f_1 = {frozenset([item]): count for item, count in counts.items() if count >= min_support}

    k = 2
    f_k_minus_1 = f_1

    # F_1 is already frequent
    frequent_itemsets = [items_count_pair for items_count_pair in f_1.items()]
    
    while f_k_minus_1:
        candidates = set()
        f_k_minus_1_keys = sorted(list(f_k_minus_1.keys()), key=lambda s: sorted(s))
        
        # Candidate generation using F_k-1 x F_k-1
        for i in range(len(f_k_minus_1_keys)):
            for j in range(i + 1, len(f_k_minus_1_keys)):
                l1 = sorted(list(f_k_minus_1_keys[i]))
                l2 = sorted(list(f_k_minus_1_keys[j]))
                if l1[:-1] == l2[:-1]:  # join condition
                    candidate = f_k_minus_1_keys[i] | f_k_minus_1_keys[j]
                    if len(candidate) == k:
                        candidates.add(candidate)
        
        # Count candidates
        candidate_counts = {c: 0 for c in candidates}
        for entry in parsed_entries:
            for candidate in candidates:
                if candidate.issubset(entry):
                    candidate_counts[candidate] += 1
        
        # Prune
        f_k = {itemset: count for itemset, count in candidate_counts.items() if count >= min_support}
        
        if not f_k:
            break
        
        
        frequent_itemsets.extend([items_count_pair for items_count_pair in f_k.items()])
        f_k_minus_1 = f_k
        k += 1
    
    return frequent_itemsets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apriori algorithm")
    parser.add_argument("--input", type=str, default="data.csv", help="Input file path, relative to HW2 root directory")
    parser.add_argument("--min_support", type=int, default=500, help="Minimum (absolute) support (integer)")
    parser.add_argument("--colname", type=str, default="text_keywords", help="Column name in the input file")
    parser.add_argument("--output", type=str, default="output.txt", help="Output file name. Will be outputted to the HW2 root directory")
    args = parser.parse_args()

    # Load data
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_path = os.path.join(script_path, args.input)
    df = pd.read_csv(data_path)

    # Run apriori
    frequent_itemsets = apriori(df, args.colname, args.min_support)

    # Output
    output_path = os.path.join(script_path, args.output)

    with open(output_path, "w") as f:
        for items_count_pair in frequent_itemsets:
            line = " ".join(list(items_count_pair[0])) + f" ({items_count_pair[1]})\n"
            print(line.strip())
            f.write(line)