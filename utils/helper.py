import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, GlobalAveragePooling2D, Lambda, RandomFlip, RandomTranslation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import VGG16, InceptionResNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

from tensorflow.random import set_seed
from utils import constants


np.random.seed(constants.SEED)
set_seed(constants.SEED)


def create_dataset(dataset_path,
                    image_size=constants.IMAGE_SIZE,
                    batch_size=constants.BATCH_SIZE):
    """
    Crea un conjunto de datos a partir de un directorio de imágenes.

    Args:
        dataset_path (str): La ruta al directorio que contiene las imágenes.
        image_size (int, opcional): El tamaño al que se redimensionarán las imágenes. De forma predeterminada, es constants.IMAGE_SIZE.
        batch_size (int, opcional): El tamaño de lote para el conjunto de datos. De forma predeterminada, es constants.BATCH_SIZE.

    Returns:
        tf.data.Dataset: El conjunto de datos creado.
    """
    label_mode = 'categorical'
    dataset = image_dataset_from_directory(dataset_path,
                                           image_size=image_size,
                                           batch_size=batch_size,
                                           label_mode=label_mode
                                          )

    return dataset


def get_batch_info(dataset, full_view=True):
    """
    Obtiene información sobre el tamaño y contenido de un batch del conjunto de datos.

    Args:
        dataset (tf.data.Dataset): El conjunto de datos del que se quiere obtener la información del batch.
        full_view (bool, opcional): Si es True, se imprime la información completa del batch. De forma predeterminada, es True.

    Returns:
        None
    """
    for X_batch, y_batch in dataset:
        print(f"Tamaño del batch de entrada: {X_batch.shape}")
        print(f"Tamaño del batch de salida: {y_batch.shape}")
        
        if full_view == True:
            print(f"Batch de entrada:\n{X_batch}")
            print(f"Batch de salida:\n{y_batch}")
        break


def create_model(optimizer = 'sgd'):
    """
    Crea un modelo de aprendizaje profundo de Keras para clasificar imágenes.

    Args:
        optimizer

    Returns:
        Un modelo Keras compilado.

    El modelo consta de las siguientes capas:
    1. Capa de escalado (Rescaling)
    2. Capa de convolución (Conv2D)
    3. Capa de aplanamiento (Flatten)
    4. Capa completamente conectada (Dense)

    La función de pérdida utilizada es 'categorical_crossentropy', 
    que es adecuada para problemas de clasificación multiclase. 

    """
    model = Sequential()
    model.add(Rescaling(scale=1./255, input_shape=(constants.IMAGE_SIZE[0],constants.IMAGE_SIZE[1], constants.IMAGE_CHANNELS)))
    model.add(Conv2D(constants.NUM_CONVOLUTIONAL_FILTERS, constants.CONVOLUTIONAL_KERNEL_SIZE, activation='linear'))
    model.add(Flatten())
    model.add(Dense(constants.NUM_DENSE_UNITS, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def create_model_3():
    """
    Crea un modelo de aprendizaje profundo de Keras para clasificar imágenes.

    Returns:
        Un modelo Keras compilado.

    El modelo consta de las siguientes capas:
    1. Capa de escalado (Rescaling)
    2. Capa de convolución (Conv2D)
    3. Capa de pooling
    4. Capa de aplanamiento
    5. Capa completamente conectada

    La función de pérdida utilizada es 'categorical_crossentropy'
    El optimizador utilizado es 'adam', que es un optimizador de gradiente estocástico adaptativo.
    """
    model = Sequential()
    model.add(Rescaling(scale=constants.SCALE, input_shape=(constants.IMAGE_SIZE[0], constants.IMAGE_SIZE[1], constants.IMAGE_CHANNELS)))
    model.add(Conv2D(constants.NUM_CONVOLUTIONAL_FILTERS, constants.CONVOLUTIONAL_KERNEL_SIZE, activation='linear'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(constants.NUM_DENSE_UNITS, activation='sigmoid'))


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])  # Conitnuamos con el optimizador a 'adam'
    return model

def create_model_4():
    """
    Crea un modelo de aprendizaje profundo de Keras para clasificar imágenes.

    Returns:
        Un modelo Keras compilado.

    Este modelo está diseñado para clasificar imágenes en función de sus características. La función crea un modelo de aprendizaje profundo de Keras y lo compila con una función de pérdida y un optimizador específicos.

    El modelo consta de las siguientes capas:
    1. Capa de escalado: escala los valores de píxeles de las imágenes de entrada entre 0 y 1.
    2. Capa de convolución: aplica un filtro de convolución a la imagen de entrada para extraer características.
    3. Capa de pooling: reduce la dimensionalidad de la salida de la capa de convolución para reducir la cantidad de parámetros.
    4. Capa de convolución: aplica otro filtro de convolución a la imagen de entrada para extraer características más complejas.
    5. Capa de pooling: reduce la dimensionalidad de la salida de la capa de convolución para reducir la cantidad de parámetros.
    6. Capa de aplanamiento: aplana la salida de la capa de pooling para prepararla para la capa densa.
    7. Capa completamente conectada: clasifica las características aplanadas en una de las clases posibles.
    8. Capa de dropout: aplica regularización para evitar sobreajuste.

    La función de pérdida utilizada es 'categorical_crossentropy', que es adecuada para problemas de clasificación multiclase. El optimizador utilizado es 'adam', que es un optimizador de gradiente estocástico adaptativo.
    """
    model= Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(constants.IMAGE_SIZE[0], constants.IMAGE_SIZE[1], constants.IMAGE_CHANNELS)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Capa de salida
    model.add(Dense(constants.NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    return model


def create_image_augmentor():
    """
     Crea un image augmentor con random rotation, shift, shear, zoom, and flip.

     Returns:
         datagen (ImageDataGenerator): La image augmentor.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen


def create_early_stopping():
    """
    Crea unn early stopping callback con patience 3 y restore con los mejores pesos.

    Returns:
        early_stopping (EarlyStopping): The early stopping callback.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    return early_stopping


def create_numpy_matrices(dataset):
    """
    Crea numpy matrices desde un dataset.

    Args:
        dataset (dataset)

    Returns:
        images (numpy array)
        labels (numpy array)
    """
    images = np.array([image for image, label in dataset])
    labels = np.array([label for image, label in dataset])
    return images, labels


def extract_numpy_matrices(dataset):
    """
    Extrae numpy matrices desde un dataset de TensorFlow.

    Args:
        dataset (tf.data.Dataset)

    Returns:
        images (numpy array): The images matrix.
        labels (numpy array): The labels matrix.
    """
    images_list = []
    labels_list = []

    for images, labels in dataset:
        images_list.append(images.numpy())
        labels_list.append(labels.numpy())

    images = np.concatenate(images_list)
    labels = np.concatenate(labels_list)

    return images, labels


def plot_loss_and_accuracy(loss_values, val_loss_values, acc_values, val_acc_values):
    """
    Plotea training and validation loss and accuracy.

    Args:
        loss_values (list): Lista de training loss values.
        val_loss_values (list): Lista de  validation loss values.
        acc_values (list): Lista de  training accuracy values.
        val_acc_values (list):Lista de validation accuracy values.
    """
    epochs_x = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_x, loss_values, 'bo', label='Training loss')
    plt.plot(epochs_x, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(epochs_x, acc_values, 'bo', label='Training acc')
    plt.plot(epochs_x, val_acc_values, 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()

    plt.show()



def create_vgg16_transfer_learning_model(input_shape=(256, 256, 3), num_classes=constants.NUM_CLASSES):
    """
    Crea un modelo de aprendizaje profundo de transferencia utilizando la red VGG16 pre-entrenada.

    Args:
        input_shape (tuple): La forma de entrada de la imagen (alto, ancho, canales). Por defecto es (256, 256, 3).
        num_classes (int): El número de clases para la clasificación. Por defecto es 6.

    Returns:
        Un modelo Keras compilado.

    Este modelo utiliza la red VGG16 pre-entrenada como base y agrega capas superiores para la clasificación. 
    La función congela las capas convolucionales de la red base para que no se actualicen durante el entrenamiento y agrega una capa de pooling global y capas densas superiores para la clasificación.

    La función de pérdida utilizada es 'categorical_crossentropy'
    El optimizador utilizado es 'adam', que es un optimizador de gradiente estocástico adaptativo.

    Además, se aplica data augmentation para aumentar la variedad de las imágenes de entrenamiento y reducir el sobreajuste.
    """
    vgg16_model = VGG16(include_top=False, input_shape=input_shape)
    vgg16_model.trainable = False

  
    model_vgg16 = Sequential()
    model_vgg16.add(Lambda(preprocess_input, input_shape=input_shape))
    
    # Aumentacion de datos
    model_vgg16.add(RandomFlip(mode="horizontal"))
    model_vgg16.add(RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)))
    
    model_vgg16.add(vgg16_model)
    model_vgg16.add(GlobalAveragePooling2D())
    model_vgg16.add(Dense(256, activation='relu'))
    model_vgg16.add(Dropout(0.6))
    model_vgg16.add(Dense(128, activation='relu'))
    model_vgg16.add(Dropout(0.5))
    model_vgg16.add(Dense(num_classes, activation='softmax'))

    model_vgg16.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    
    return model_vgg16


def create_inceptionresnetv2_transfer_learning_model(input_shape=(256, 256, 3), num_classes=6):
    """
        Crea un modelo de aprendizaje de transferencia InceptionResNetV2 con capas personalizadas.

        Args:
            input_shape: Por defecto a (256, 256, 3).
            num_classes: Por defecto a 6.

        Returns:
            Un modelo InceptionResNetV2 de aprendizaje de transferencia compilado.
    """
    inception = InceptionResNetV2(include_top=False, input_shape=input_shape)
    inception.trainable = False # Congela las capas pre-entrenadas

    model_inception = Sequential()
    model_inception.add(Lambda(preprocess_input, input_shape=input_shape))
    model_inception.add(RandomFlip(mode="horizontal"))
    model_inception.add(RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)))

    model_inception.add(inception)
    model_inception.add(GlobalAveragePooling2D())
    
    model_inception.add(Dense(512, activation='relu'))
    model_inception.add(Dropout(0.6))
    model_inception.add(Dense(128, activation='relu'))
    model_inception.add(Dropout(0.5))
    
    model_inception.add(Dense(num_classes, activation='softmax'))

    model_inception.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    
    return model_inception

def plot_accuracy(model_accuracy_dict):
    """
    Crea un gráfico de barras que muestra la precisión de cada modelo hecho.

    Args:
        model_accuracy_dict (dict): Un diccionario que mapea los nombres de los modelos a sus respectivas precisiones.

    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_accuracy_dict.keys(), model_accuracy_dict.values(), color='skyblue', edgecolor='black')

    plt.ylabel('Accuracy lograda', fontsize=12)
    plt.title('Precisión de cada modelo', fontsize=12, fontweight='bold')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
    plt.ylim(0, 1) 
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

