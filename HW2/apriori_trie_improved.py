import argparse
import pandas as pd
import os
import sys
import pickle


class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    # Insert and count
    def insert(self, itemset):
        node = self.root
        for item in sorted(itemset):
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]
        node.count += 1

    # Dfreq_set the trie to get frequent itemsets
    def get_frequent_itemsets(self, min_support, current_prefix=frozenset(), node=None):
        if node is None:
            node = self.root

        frequent_itemsets = []
        if current_prefix and node.count >= min_support:
            frequent_itemsets.append((current_prefix, node.count))

        for item, child in node.children.items():
            frequent_itemsets.extend(
                self.get_frequent_itemsets(min_support, current_prefix | frozenset([item]), child)
            )
        return frequent_itemsets


def parse_entry(entry) -> frozenset:
    return frozenset(entry.split(";"))

def apriori_trie(df, colname, min_support):
    comparison_count = 0
    subset_check_count = 0
    trie_insert_count = 0
    op_count = 0
    entries = df[colname].dropna()
    parsed_entries = [parse_entry(entry) for entry in entries]

    trie = Trie()

    # F1
    for entry in parsed_entries:
        for item in entry:
            trie_insert_count += 1
            trie.insert(frozenset([item]))
    frequent_itemsets = trie.get_frequent_itemsets(min_support)

    current_frequents = {freq_set for freq_set, _ in frequent_itemsets if len(freq_set) == 1}
    k = 2

    while current_frequents:
        candidates = set()
        current_list = sorted(list(current_frequents), key=lambda s: sorted(s)) # Sort itemset for trie optimization
        
        # Candidate generation using F_k-1 x F_k-1
        for i in range(len(current_list)):
            for j in range(i+1, len(current_list)):
                comparison_count += 1 # Count comparison
                l1 = sorted(current_list[i])
                l2 = sorted(current_list[j])
                if l1[:-1] == l2[:-1]:
                    candidate = current_list[i] | current_list[j]
                    if len(candidate) == k:
                        candidates.add(candidate)
        if not candidates:
            break
        
        # Prune
        candidate_trie = Trie()
        for entry in parsed_entries:
            for candidate in candidates:
                subset_check_count += 1 # Count subset check
                if candidate.issubset(entry):
                    candidate_trie.insert(candidate)
        
        freq_candidates = candidate_trie.get_frequent_itemsets(min_support)
        new_frequents = {freq_set for freq_set, _ in freq_candidates if len(freq_set) == k}
        
        if not new_frequents:
            break
        

        frequent_itemsets.extend([(freq_set, cnt) for freq_set, cnt in freq_candidates if len(freq_set)==k])
        current_frequents = new_frequents
        k += 1

    # print(f"Comparison count: {comparison_count}")
    # print(f"Subset check count: {subset_check_count}")
    # print(f"Trie insert count: {trie_insert_count}")
    return frequent_itemsets, comparison_count, subset_check_count, trie_insert_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apriori algorithm using Trie: Improved version")
    parser.add_argument("--input", type=str, default="data.csv", help="Input file path")
    parser.add_argument("--min_support", type=int, default=500, help="Minimum support count")
    parser.add_argument("--colname", type=str, default="text_keywords", help="Column name for transactions")
    parser.add_argument("--output", type=str, default="output.txt", help="Output file name")
    parser.add_argument("--pickle", type=bool, default=False, help="Option to pickle results")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose output")
    args = parser.parse_args()

    # Load data
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_path = os.path.join(script_path, args.input)
    df = pd.read_csv(data_path)

    # Run Apriori with trie optimization
    result, comparison_count, subset_check_count, trie_insert_count = apriori_trie(df, args.colname, args.min_support)
    result.sort(key=lambda x: (x[1], list(x[0])))
    
    # Write output
    output_path = os.path.join(script_path, args.output)
    with open(output_path, "w") as fout:
        for itemset, count in result:
            line = " ".join(list(itemset)) + f" ({count})\n"
            if args.verbose:
                print(line.strip())
            fout.write(line)
    
    if args.pickle:
        pickle_file = os.path.join(script_path, "apriori_trie_improved.pickle")
        with open(pickle_file, "wb") as fout:
            pickle.dump(result, fout)
