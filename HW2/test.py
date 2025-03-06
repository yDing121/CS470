o1 = set()
so = set()

with open("output(1).txt", "r") as fin:
    for line in fin:
        o1.add(frozenset(line.split()))

with open("std_output.txt", "r") as fin:
    for line in fin:
        so.add(frozenset(line.split()))


print(f"The sets are equivalent:\t{o1 == so}")
