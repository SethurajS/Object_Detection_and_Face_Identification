
"""""""""""""""""""""""""""""""""""""""""""""IMPORTING THE REQUIREMENTS"""""""""""""""""""""""""""""""""""""""""""""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt


""""""""""""""""""""""""""""""""""""""""""""""""" TRAINING THE MODEL """""""""""""""""""""""""""""""""""""""""""""""""

IMG_SIZE = 160


def training():

    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    MobileNetModel = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')  # MODEL -- MobileNetV2 trained on ImageNet

    MobileNetModel.trainable = False

    folders = glob('Datasets/Train/*')

    x = Flatten()(MobileNetModel.output)

    prediction = Dense(len(folders), activation='softmax')(x)

    model = Model(inputs=MobileNetModel.input, outputs=prediction)

    model.summary()

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
        metrics=['accuracy']
    )



    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    training_data = train_datagen.flow_from_directory('D:\Object_detection\Datasets\Train',
                                                     target_size=(160, 160),
                                                     class_mode='sparse')
    classes = training_data.class_indices
    print(classes)

    validation_data = validation_datagen.flow_from_directory('D:\Object_detection\Datasets\Test',
                                                target_size=(160, 160),
                                                class_mode='sparse')

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=5)

    # LOSS
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    plt.savefig('LossVal_loss')

    # ACCURACY
    plt.plot(history.history['acc'], label='train acc')
    plt.plot(history.history['val_acc'], label='val acc')
    plt.legend()
    plt.show()
    plt.savefig('AccVal_acc')

    model.save('D:\Object_detection\model.h5')

    print("MODEL UPDATED")

    return classes  # FOR LABELING PREDICTION

