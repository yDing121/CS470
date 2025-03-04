import argparse
import pandas as pd
import os
import sys
import pickle


def parse_entry(entry) -> frozenset:
    return frozenset(entry.split(";"))


def apriori(df, column_name, min_support):
    comparison_count = 0
    subset_check_count = 0
    
    # Drop entries with empty item sets
    entries = df[column_name].dropna()

    # Immutable sets for hashing
    parsed_entries = [parse_entry(entry) for entry in entries]

    # Get item counts
    counts = {}
    for entry in parsed_entries:
        for item in entry:
            comparison_count += 1
            counts[item] = counts.get(item, 0) + 1
    
    # F_1
    f_1 = {frozenset([item]): count for item, count in counts.items() if count >= min_support}

    k = 2
    f_k_minus_1 = f_1

    # F_1 is already frequent
    frequent_itemsets = list(f_1.items())
    
    while f_k_minus_1:
        candidates = set()
        f_k_minus_1_keys = sorted(f_k_minus_1.keys(), key=lambda s: sorted(s))
        
        # Candidate generation using F_k-1 x F_k-1
        for i in range(len(f_k_minus_1_keys)):
            for j in range(i + 1, len(f_k_minus_1_keys)):
                l1 = sorted(f_k_minus_1_keys[i])
                l2 = sorted(f_k_minus_1_keys[j])
                comparison_count += 1 # Count comparison
                if l1[:-1] == l2[:-1]:  # join condition
                    candidate = f_k_minus_1_keys[i] | f_k_minus_1_keys[j]
                    if len(candidate) == k:
                        candidates.add(candidate)
        
        # Count candidates
        candidate_counts = {c: 0 for c in candidates}
        for entry in parsed_entries:
            for candidate in candidates:
                subset_check_count += 1 # Count subset check
                if candidate.issubset(entry):
                    candidate_counts[candidate] += 1
        
        # Prune
        f_k = {itemset: count for itemset, count in candidate_counts.items() if count >= min_support}
        
        if not f_k:
            break
        
        frequent_itemsets.extend(f_k.items())
        f_k_minus_1 = f_k
        k += 1
    
    # print(f"Comparison count: {comparison_count}")
    # print(f"Subset check count: {subset_check_count}")
    return frequent_itemsets, comparison_count, subset_check_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apriori algorithm")
    parser.add_argument("--input", type=str, default="data.csv", help="Input file path, relative to HW2 root directory")
    parser.add_argument("--min_support", type=int, default=500, help="Minimum (absolute) support (integer)")
    parser.add_argument("--colname", type=str, default="text_keywords", help="Column name in the input file")
    parser.add_argument("--output", type=str, default="output.txt", help="Output file name. Will be outputted to the HW2 root directory")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose output")
    args = parser.parse_args()

    # Load data
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_path = os.path.join(script_path, args.input)
    df = pd.read_csv(data_path)

    # Run apriori
    frequent_itemsets, comparison_count, subset_check_count = apriori(df, args.colname, args.min_support)

    # Semi-radix sort for ordering primarily by count, then by items
    frequent_itemsets.sort(key = lambda x: x[0])
    frequent_itemsets.sort(key = lambda x: x[1])
    
    # Output
    output_path = os.path.join(script_path, args.output)

    with open(output_path, "w") as fout:
        for items_count_pair in frequent_itemsets:
            line = " ".join(list(items_count_pair[0])) + f" ({items_count_pair[1]})\n"
            fout.write(line)
    
    with open(f"{args.algorithm}.pickle", "wb") as fout:
        pickle.dump(frequent_itemsets, fout)