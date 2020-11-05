import gzip
import numpy as np

#ogni immagine e' composta da 784 byte (28x28 pixel)
#ciascun byte rappresenta un valore da 0 a 255
#(che identificano il colore in scala di grigi di quel pixel)
IMAGE_SIZE = 784

#caricamento dei dati
def load_data(num_images):
    #unpack del file .gz delle immagini
    with gzip.open('./trainingdata/training_set_images.gz', 'r') as file_training_set:
        #lettura dei primi 16 byte
        buffer = file_training_set.read(16) #lettura di magic number e dimensioni
        training_set = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        #lettura dei successivi 784 * num_images byte
        buffer = file_training_set.read(num_images*IMAGE_SIZE) #lettura delle immagini
        #inserimento e modellamento dei byte in una matrice di num_images righe
        #e 784 colonne (ciascuna riga rappresenta un'immagine)
        training_set = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        training_set = training_set.reshape(num_images, IMAGE_SIZE)
        for i, row in enumerate(training_set):
            for j, col in enumerate(row):
                training_set[i][j] = training_set[i][j]/255 #trasformo ciascun pixel in un numero da 0 a 1
        ret_set = []
        for row in training_set:
            ret_set.append(np.reshape(row, (784, 1)))
    #unpack del file .gz delle labels
    with gzip.open('./trainingdata/trainin_set_labels.gz') as file_labels:
        #lettura dei primi 8 byte
        buf2 = file_labels.read(8)
        #lettura dei successivi num_images byte
        buf2 = file_labels.read(num_images)
        #modellamento dei byte in un vettore di num_images elementi
        lab_temp = np.frombuffer(buf2, dtype=np.uint8).astype(np.int64)
        #trasformazione delle label da numeri interi a codifica "one-hot"
        labels = [vectorized_result(lab) for lab in lab_temp]
    data = [(input, target) for input, target in zip(ret_set, labels)]
    return data

#vettorizzazione del risultato
def vectorized_result(index):
    #vettori di 10 posizioni
    vector = np.zeros((10, 1))
    #l'unico valore diversa da 0 e' quello della label
    vector[index] = 1
    return vector

def check_distribution(data):
    #calcola il numero di elementi appartenenti ad ogni classe.
    res = np.zeros(10)
    for val, label in data:
        res[np.argmax(label)] += 1
    return res

if __name__ == "__main__":
    data = load_data(3000)
    tr_D = data[:2000]
    val = data[2000:2500]
    test = data[2500:3000]
    res = check_distribution(tr_D)
    print(res)
    res = check_distribution(val)
    print(res)
    res = check_distribution(test)
    print(res)

