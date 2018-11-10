import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image

from Utility import get_nb_files, get_data, plot_training, resize_image, plot_preds

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam, sgd
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image

FC_SIZE = 1024
NB_VGG_LAYERS_TO_FREEZE = 20
IM_WIDTH, IM_HEIGHT = 256, 256

def setup_to_transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=adam(), loss='binary_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu', kernel_initializer='he_normal')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def setup_to_finetune(model):
    for layer in model.layers[:NB_VGG_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_VGG_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=sgd(lr=1e-5, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

def train(args):
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    batch_size = int(args.batch_size)


    # MARK :- prepare train data generator

    train_datagen = ImageDataGenerator(
        preprocessing_function=None,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        featurewise_center=args.featurewise_center
    )

    # MARK :- fit train data generator for featurewise_center

    if args.featurewise_center:
        train_x = get_data(args.train_dir, tar_size=(IM_WIDTH, IM_HEIGHT, 3))
        train_datagen.fit(train_x / 225)


    # MARK :- prepare valid data generator

    valid_datagen = ImageDataGenerator(
        preprocessing_function=None,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        featurewise_center=args.featurewise_center
    )

    # MARK :- fit valid data generator for featurewise_center

    if args.featurewise_center:
        valid_x = get_data(args.val_dir, tar_size=(IM_WIDTH, IM_HEIGHT, 3))
        valid_datagen.fit(valid_x / 225)


    # MARK :- prepare train and valid generators

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = valid_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical'
    )


    # MARK :- prepare base model

    base_model = VGG16(
        weights='imagenet',
        include_top=False, classes=2,
        input_shape=(IM_WIDTH, IM_HEIGHT, 3)
    )


    # MARK :- setup model to transfer learning

    model = add_new_last_layer(base_model, nb_classes)
    setup_to_transfer_learn(model, base_model)


    # MARK :- prepare VA and VL checkpoints for transfer learning

    best_tl_va = ModelCheckpoint(
        'Models/tl_va_' + args.output_model_file,
        monitor='val_acc',
        mode='max',
        verbose=1,
        save_best_only=True
    )

    best_tl_vl = ModelCheckpoint(
        'Models/tl_vl_' + args.output_model_file,
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )


    # MARK :- fit model with transfer learning

    history_tl = model.fit_generator(
        train_generator,
        steps_per_epoch=int(round(nb_train_samples / batch_size)),
        epochs=int(args.tl_epoch),
        validation_data=validation_generator,
        validation_steps=int(round(nb_val_samples / batch_size)),
        class_weight='auto',
        callbacks=[best_tl_va, best_tl_vl]
    )

    plot_training(history_tl, 'tl_history.png')


    # MARK :- load best transfer learning model and setup it

    model = load_model(filepath=args.tl_model)
    setup_to_finetune(model)


    # MARK :- prepare VA and VL checkpoints for fine tuning

    best_ft_va = ModelCheckpoint(
        'Models/ft_va_' + args.output_model_file,
        monitor='val_acc',
        mode='max',
        verbose=1,
        save_best_only=True
    )

    best_ft_vl = ModelCheckpoint(
        'Models/ft_vl_' + args.output_model_file,
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )


    # MARK :- fit model with fine tuning

    history_ft = model.fit_generator(
        train_generator,
        steps_per_epoch=int(round(nb_train_samples / batch_size)),
        epochs=int(args.ft_epoch),
        validation_data=validation_generator,
        validation_steps=int(round(nb_val_samples / batch_size)),
        class_weight='auto',
        callbacks=[best_ft_va, best_ft_vl]
    )

    plot_training(history_ft, 'ft_history.png')


if __name__ == "__main__":

    a = argparse.ArgumentParser()

    a.add_argument("--train_dir", default='Dataset/Train')
    a.add_argument("--val_dir", default='Dataset/Validation')
    a.add_argument("--tl_epoch", default=15)
    a.add_argument("--ft_epoch", default=5)
    a.add_argument("--batch_size", default=30)
    a.add_argument("--output_model_file", default="vgg16.h5")
    a.add_argument("--image", help="path to image")
    a.add_argument("--ft_model", default="Models/ft_vl_vgg16.h5")
    a.add_argument("--tl_model", default="Models/tl_vl_vgg16.h5")
    a.add_argument("--featurewise_center", default=False)

    args = a.parse_args()

    if args.image is not None:
        model = load_model(filepath=args.ft_model)
        img = resize_image(Image.open(args.image), (IM_WIDTH, IM_HEIGHT))
        x = np.expand_dims(image.img_to_array(img), axis=0)
        # x = preprocess_input(x)
        preds = model.predict(x)
        plot_preds(Image.open(args.image), preds[0])
        sys.exit(1)

    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)
