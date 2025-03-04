k1_km1 = set()
km1_km1 = set()
trie = set()

with open("HW2/k-1k-1.txt", "r") as fin:
    for line in fin:
        k1_km1.add(frozenset(line.split()))

with open("HW2/k1k-1.txt", "r") as fin:
    for line in fin:
        km1_km1.add(frozenset(line.split()))

with open("HW2/trie.txt", "r") as fin:
    for line in fin:
        trie.add(frozenset(line.split()))

print(k1_km1 == km1_km1 and km1_km1 == trie)
# print(k1_km1 - km1_km1)
# print(km1_km1 - k1_km1)