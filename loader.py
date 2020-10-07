import gzip
import numpy as np
import matplotlib.pyplot as plt

with gzip.open('./trainingdata/training_set_images.gz', 'r') as file:
    buf = file.read(16)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    image_size = 784
    buf = file.read(3*image_size) # letto un'immagine intera
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(3, 28, 28, 1)
    image = np.asarray(data[0]).squeeze()
    plt.imshow(image)
    plt.show()
    lab = gzip.open('./trainingdata/trainin_set_labels.gz')
    buf2 = lab.read(8)
    buf2 = lab.read(1)
    label = np.frombuffer(buf2, dtype=np.uint8).astype(np.int64)
    print(label)