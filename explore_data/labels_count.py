import numpy as np
def load_labels(labelset,data_size):
    labels = []
    with open(labelset) as f:
        lines = f.readlines()
        for item in lines[:data_size]:
            labels.append(int(item.split("\n")[0]))
    labels = np.asarray(labels)
    return labels

file_name = "testing_label.txt"
data_size = 300000
labels = load_labels(file_name,data_size)

#show the class distribution of a dataset
print("ture",labels.sum())
print("false",data_size-labels.sum())