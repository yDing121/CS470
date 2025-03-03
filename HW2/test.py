k1_km1 = set()
km1_km1 = set()

with open("HW2/k-1k-1.txt", "r") as fin:
    for line in fin:
        k1_km1.add(frozenset(line.split()))

with open("HW2/k1k-1.txt", "r") as fin: #open the second file with a new file object
    for line in fin:
        km1_km1.add(frozenset(line.split()))

print(k1_km1 == km1_km1)
print(k1_km1 - km1_km1)
print(km1_km1 - k1_km1)