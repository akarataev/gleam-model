import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array


def get_data(DIR_PATH, tar_size):
    x_data = []
    y_data = []

    class_to_index = {
        name: index for index, name in
        enumerate(os.listdir(DIR_PATH))
    }

    for p_dir in os.listdir(DIR_PATH):

        y_vec = np.array(
            [0 if i != class_to_index[p_dir] else 1
             for i in range(len(class_to_index))]
        )

        for f in os.listdir(os.path.join(DIR_PATH, p_dir)):
            img = load_img(os.path.join(DIR_PATH, p_dir, f), target_size=tar_size)
            img_arr = img_to_array(img)
            x_data.append(img_arr)
            y_data.append(y_vec)

    return np.array(x_data), np.array(y_data)


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


def plot_auc_roc(tpr, fpr, auc, name):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=name+' (area = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC/ROC curve')
    plt.legend(loc='best')
    plt.savefig('Models/Plots/' + name + '.png')
    plt.show()

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
