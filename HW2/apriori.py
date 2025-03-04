import apriori_k1km1
import apriori_km1km1
import apriori_trie_improved
import sys
import os
import argparse
import pandas as pd
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apriori algorithm")
    parser.add_argument("--input", type=str, default="data.csv", help="Input file path, relative to HW2 root directory")
    parser.add_argument("--min_support", type=int, default=500, help="Minimum (absolute) support (integer)")
    parser.add_argument("--colname", type=str, default="text_keywords", help="Column name in the input file")
    parser.add_argument("--output", type=str, default="output.txt", help="Output file name. Will be outputted to the HW2 root directory")
    parser.add_argument("--pickle", type=bool, default="False", help="Whether to pickle the itemsets")
    parser.add_argument("--algorithm", type=str, default="km1km1", help="Apriori variant")
    args = parser.parse_args()

    # Load data
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_path = os.path.join(script_path, args.input)
    df = pd.read_csv(data_path)

    # Select algorithm
    mapping = {
        "k1km1": apriori_k1km1.apriori,
        "km1km1": apriori_km1km1.apriori,
        "trie": apriori_trie_improved.apriori
    }

    try:
        apriori = mapping[args.algorithm]
    except KeyError:
        print("Invalid algorithm")
        sys.exit(1)
    
    # Run apriori
    frequent_itemsets, *_ = apriori(df, args.colname, args.min_support)

    # Semi-radix sort for ordering primarily by count, then by items
    frequent_itemsets.sort(key=lambda x: (x[1], list(x[0])))
    
    # Output
    output_path = os.path.join(script_path, args.output)

    with open(output_path, "w") as fout:
        for items_count_pair in frequent_itemsets:
            line = " ".join(list(items_count_pair[0])) + f" ({items_count_pair[1]})\n"
            fout.write(line)
    print(f"Output for {args.algorithm} saved to {output_path}")
    
    with open("k1k-1.pickle", "wb") as fout:
        pickle.dump(frequent_itemsets, fout)