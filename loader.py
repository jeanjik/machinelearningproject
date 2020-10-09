import gzip
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 784


def load_data(path, num_images):
    with gzip.open('./trainingdata/training_set_images.gz', 'r') as file_training_set:
        buffer = file_training_set.read(16)
        training_set = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        buffer = file_training_set.read(num_images*IMAGE_SIZE) # letto un'immagine intera
        training_set = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        training_set = training_set.reshape(num_images, IMAGE_SIZE)
        for i, row in enumerate(training_set):
            for j, col in enumerate(row):
                training_set[i][j] = training_set[i][j]/255
    with gzip.open('./trainingdata/trainin_set_labels.gz') as file_labels:
        buf2 = file_labels.read(8)
        buf2 = file_labels.read(num_images)
        lab_temp = np.frombuffer(buf2, dtype=np.uint8).astype(np.int64)
        labels = [vectorized_result(lab) for lab in lab_temp]
    data = [(input, target) for input, target in zip(training_set, labels)]
    return data

def vectorized_result(index):
    vector = np.zeros(10)
    vector[index] = 1
    return vector



nn = load_data(0, 3)
for input, target in nn:
    print(input)
    print(target)