import numpy as np

train_file = "./data/data_length.txt"

length = []

with open(train_file, mode="r", encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        length.append(int(line))

max_len = max(length)
print(max_len)

import matplotlib.pyplot as plt

plt.hist(length, bins=max_len)
plt.show()