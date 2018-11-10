import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array


def get_data(DIR_PATH, tar_size):
    data = []
    for p_dir in os.listdir(DIR_PATH):
        for f in os.listdir(os.path.join(DIR_PATH, p_dir)):
            img = load_img(os.path.join(DIR_PATH, p_dir, f), target_size=tar_size)
            img_arr = img_to_array(img)
            data.append(img_arr)
    return np.array(data)


def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def resize_image(img, target_size):
    if img.size != target_size:
        img = img.resize(target_size)
    return img


def plot_training(history, file):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure()
    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.savefig("Models/Plots/va_" + file)

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.savefig("Models/Plots/vl_" + file)




def plot_preds(image, preds):
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  labels = ("Benign", "Malignant")
  print(preds)
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()
