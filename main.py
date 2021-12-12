import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
### CNN models ###
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.models import Sequential, save_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.utils import np_utils
from keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.layers import BatchNormalization
from keras import models
from keras.utils.vis_utils import plot_model
from keras.layers import Input, GlobalAveragePooling2D,concatenate
from keras.models import Model
from keras.models import model_from_json
from tensorflow.keras import layers
from keras.applications.inception_v3 import InceptionV3


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score



def define_consts():
    batch_size = 16
    num_epochs = 50
    input_shape = (48, 48, 1)
    validation_split = .2
    verbose = 1
    num_classes = 7
    base_path = 'models/'
    image_size=(48,48)
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    result = ['Very unlikely', 'Unlikely', 'Possible', 'Likely', 'Very likely']
    floor = [(0, 0.19), (0.2, 0.39), (0.4, 0.59), (0.6, 0.79), (0.8, 1)]
    return batch_size, num_epochs, input_shape, validation_split, verbose, num_classes, base_path, image_size, labels, result, floor

def read_data(image_size):
    data = pd.read_csv('./data/fer2013/fer2013.csv')
    print(len(data))
    data['pixels']=data['pixels'].astype("string")
    # print(data['pixels'])
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    # iterate through all pixels and create a matrix(face) of size 48 x 48
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.strip().split(' ',48*48)]
        if len(face) == 2304:
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
    # last = faces[-1]
    # faces.append(last)
    # del faces[-1]
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    faces /= 127.5
    faces -= 1.
    plt.figure()
    plt.imshow(faces[0]) 
    plt.show()  # display it

    emotions = pd.get_dummies(data['emotion']).to_numpy()

    print(len(faces))
    datagen = ImageDataGenerator(
            zoom_range=0.2,          # randomly zoom into images
            rotation_range=10,       # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,   # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,    # randomly flip images
            vertical_flip=False)     # randomly flip images

    xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.3,shuffle=True)
    xval,xtest,yval,ytest=train_test_split(xtest,ytest,test_size=0.3,shuffle=True)
    return xtrain, xtest, ytrain, ytest, xval, xtest, yval, ytest, datagen


def CNN():
    model = Sequential(name='CNN')
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(7))
    model.add(Activation('softmax'))
    
    return model

def setup_cnn(base_path, datagen, xtrain, ytrain, batch_size, num_epochs, xval, yval):
    CNN=CNN()

    early_stop = EarlyStopping('val_loss', patience=100)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=25, min_lr=0.00001,model='auto')
    trained_models_path = base_path + 'CNN'
    model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)
    callbacks = [model_checkpoint, early_stop, reduce_lr]

    CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    CNN_history =CNN.fit(datagen.flow(xtrain, ytrain, batch_size),
          steps_per_epoch=len(xtrain) / batch_size, 
          epochs=num_epochs, 
          verbose=1, 
          callbacks=callbacks,
          validation_data=(xval,yval))
    save_model_to_json(CNN);

def save_model_to_json(CNN):
    # serialize model to JSON
    model_json = CNN.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    CNN.save_weights("model.h5")
    print("Saved model to disk")


def get_model_from_json():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    CNN = model_from_json(loaded_model_json)
    # load weights into new model
    CNN.load_weights("model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return CNN


def test_metrics(CNN, xtest, ytest, labels, result, floor):
    ypred=CNN.predict(xtest)
    ypred_=np.argmax(ypred, axis=1)
    ytest_=np.argmax(ytest, axis=1)
    print(classification_report(ytest_, ypred_,digits=3))
    
    import itertools
    from sklearn.metrics import confusion_matrix
    from matplotlib.pyplot import figure
    
    
    fig = figure(figsize=(10, 10))
    
    label = 0
    ypred=CNN.predict(xtest)
    for p in ypred[0]:
        ind = 0
        for i in range(len(floor)):
            if p >= floor[i][0] and p <= floor[i][1]:
                ind = i
        print(labels[label], result[ind])
        label = label + 1
    rounded_predections=np.argmax(ypred, axis=1)
    rounded_labels=np.argmax(ytest, axis=1)
    cm = confusion_matrix(rounded_labels, rounded_predections)
    title='Confusion matrix '+CNN.name
    
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def test_image(CNN, image_size, floor, result, labels):
    from PIL import Image

    image = Image.open('happy1.jpg')

    size = (72,72)
    #image = image.resize(size, Image.ANTIALIAS)

    plt.figure()
    plt.imshow(image) 
    plt.show()  # display it
    pixels_image = image.getdata()
    pixels_image = list(pixels_image)

    pixels_list = []


    for tuple in pixels_image:
        gray = tuple[0] * 0.299 + tuple[1] * 0.587 + tuple[2] * 0.114
        pixels_list.append(gray)
    
    pixels_list = np.array(image)
    pixels_list = np.dot(pixels_list[...,:3], [0.2989, 0.5870, 0.1140])

    print(len(pixels_list))
    print(pixels_list)
    faces = []

    face = pixels_list
    face = np.asarray(face)
    face = cv2.resize(face.astype('uint8'),image_size)
    faces.append(face.astype('float32'))
    
    faces = np.asarray(faces)
    #faces = np.expand_dims(faces, -1)

    plt.figure()
    plt.imshow(face) 
    plt.show()  # display it

    import itertools
    from sklearn.metrics import confusion_matrix
    from matplotlib.pyplot import figure


    fig = figure(figsize=(10, 10))

    ypred=CNN.predict(faces)
    label = 0
    for p in ypred[0]:
        ind = 0
        for i in range(len(floor)):
            if p >= floor[i][0] and p <= floor[i][1]:
                ind = i
        print(labels[label], result[ind])
        label = label + 1

if __name__ == '__main__':
    batch_size, num_epochs, input_shape, validation_split, verbose, num_classes, base_path, image_size, labels, result, floor = define_consts()
    xtrain, xtest, ytrain, ytest, xval, xtest, yval, ytest, datagen = read_data(image_size)
    #setup_cnn(base_path, datagen, xtrain, ytrain, batch_size, num_epochs, xval, yval)
    CNN = get_model_from_json()
    #test_metrics(CNN, xtest, ytest, labels, result, floor)
    test_image(CNN, image_size, floor, result, labels)
    