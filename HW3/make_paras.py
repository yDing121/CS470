import random

def generate_partitions(total_entries, train_file, test_file, ratio=0.5):
    entries = list(range(1, total_entries + 1))

    random.shuffle(entries)
    split_index = int(len(entries) * ratio)
    train_set = entries[:split_index]
    test_set = entries[split_index:]
    
    with open(train_file, 'w') as f:
        for entry in train_set:
            f.write(f"{entry}\n")
    
    with open(test_file, 'w') as f:
        for entry in test_set:
            f.write(f"{entry}\n")

if __name__ == "__main__":
    total_entries = 303
    train_file = "para2_file.txt"
    test_file = "para3_file.txt"
    ratio = 0.8
    
    generate_partitions(total_entries, train_file, test_file, ratio)
