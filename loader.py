import gzip
import numpy as np


IMAGE_SIZE = 784

def load_data(num_images):
    with gzip.open('./trainingdata/training_set_images.gz', 'r') as file_training_set:
        buffer = file_training_set.read(16) # lettura di magic number e dimensioni 
        training_set = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        buffer = file_training_set.read(num_images*IMAGE_SIZE) # lettura delle immagini
        training_set = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        training_set = training_set.reshape(num_images, IMAGE_SIZE)
        for i, row in enumerate(training_set):
            for j, col in enumerate(row):
                training_set[i][j] = training_set[i][j]/255 # trasformo ciascun pixel in un numero da 0 a 1
        ret_set = []
        for row in training_set:
            ret_set.append(np.reshape(row, (784, 1)))
    with gzip.open('./trainingdata/trainin_set_labels.gz') as file_labels:
        buf2 = file_labels.read(8)
        buf2 = file_labels.read(num_images)
        lab_temp = np.frombuffer(buf2, dtype=np.uint8).astype(np.int64)
        labels = [vectorized_result(lab) for lab in lab_temp]
    data = [(input, target) for input, target in zip(ret_set, labels)]
    return data


def vectorized_result(index):
    vector = np.zeros((10, 1))
    vector[index] = 1
    return vector

def check_distribution(data):
    res = np.zeros(10)
    for val, label in data:
        res[np.argmax(label)] += 1
    return res


if __name__ == "__main__":
    data = load_data(3000)
    res = check_distribution(data)
    print(res)

