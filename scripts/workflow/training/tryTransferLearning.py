import numpy as np
import os
import sys
import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50
from keras.utils import plot_model
import random
import sqlite3

def createTransferModel(numOutputCategories):
    resnetModel = resnet50.ResNet50(weights="imagenet", include_top=False)  #Pre trained on ImageNet but without the final layer reducing to 1000 categories

    x = resnetModel.output
    x=GlobalAveragePooling2D()(x) #appears that include_top=False chops off the GlobalAveragePooling2D layer as well.

    #Toss on a new dense layer, maybe we should use more than one
    preds = Dense(numOutputCategories, activation='softmax')(x)

    transferModel = Model(inputs=resnetModel.input, outputs=preds)

    #Print the model to a file to check that we modified it as expected
    #plot_model(transferModel, to_file='transfer_model.svg');
    #print transferModel.summary();

    #let's disable training for everything prior to add_13 (look at transfer_model.svg to get a better sense of where this is, it leaves about 3 more of the repeated convolution / skip constructs)
    for layer in transferModel.layers:
        layer.trainable=False
        if layer.name == "add_13":
            break

    transferModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


    return transferModel

def listTrainingURLs(trainingDir):
    imageURLs = []
    for root,dirs,files in os.walk(trainingDir):
        for filename in files:
            if not filename.endswith(".jpg"):
                continue;
            filesize = os.path.getsize(os.path.join(root,filename))
            if filesize == 0:
                print("CORRUPT FILE: " + os.path.join(root,filename))
                continue
            imageURLs.append(os.path.join(root,filename))
    return imageURLs

def lookupTrainingOutputs(trainingURLs, dbPath):
    categories = []
    conn = sqlite3.connect(dbPath)
    cur = conn.cursor()
    for imageURL in trainingURLs:
        imgURLTuple = ("%" + imageURL[len("../output"):],)
        cur.execute("SELECT * FROM image WHERE url LIKE ?", imgURLTuple)
        rows = cur.fetchall()
        if len(rows) < 1:
            raise "COULDNT FIND ROW!"
        if len(rows) > 1:
            print("IGNORING MULTIPLE ROWS FOR URL: " + imgURLTuple[0]);
        row = rows[0]
#        print("URL: " + row[0] + " Protein: " + row[1] + " antibody: " + row[2] + " cell_line: " + row[3] + " location: " + row[4]);
        categories.append(row[4])
    conn.close()

    categoryToNumber = {}
    numberGen = 0
    for category in categories:
        if category not in categoryToNumber:
            categoryToNumber[category] = numberGen;
            numberGen += 1

    oneHots = []
    for category in categories:
        oneHot = [0] * len(categoryToNumber)
        oneHot[categoryToNumber[category]] = 1
        oneHots.append(oneHot)

#    print categories[0:5]
#    print oneHots[0:5]
    return oneHots;

def train(imageURLs, oneHots, model):
    #Each epoch will train over all of our training images in a random order with a particular batch size.  We'll mimic (or directly use, if I can figure out how) the Keras' ImageDataGenerator.

    #See full example at https://keras.io/preprocessing/image/
    x_train = imageURLs;
    y_train = oneHots;
    for e in range(5):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in customFlow(x_train, y_train, 32):
            print "Batch: " + str(batches)
            model.fit(x_batch, y_batch)
            batches += 1
            
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

def randomBatchGenerator(fileURLs, oneHots, batch_size = 32):
    indexer = range(len(fileURLs))
    batchURLs = []
    batchHots = []

    while True:
        random.shuffle(indexer)
        for randomIndex in indexer:
            batchURLs.append(fileURLs[randomIndex])
            batchHots.append(oneHots[randomIndex])
            if len(batchURLs) == batch_size:
                yield (batchURLs, batchHots)
                batchURLs = []
                batchHots= []
    
def processImageBatch(fileURLs):

    originals = []
    for fileURL in fileURLs:
        # load an image in PIL format
        original = None
        try:
            original = load_img(fileURL, target_size=(224, 224))
            originals.append(original)
        except IOError:
            raise Exception("Failed to load file: " + fileURL)

    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpyImages = []
    for original in originals:
        numpy_image = img_to_array(original)
        numpyImages.append(numpy_image)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.stack(numpyImages, axis=0)

    # prepare the image for the ResNet processing
    processed_images = resnet50.preprocess_input(image_batch.copy())

    return processed_images

def customFlow(x_train, y_train, batch_size=32):
    while True:
        for x_batch, y_batch in randomBatchGenerator(x_train, y_train, batch_size):
            processed_x_batch = processImageBatch(x_batch)
            processed_y_batch = np.stack(y_batch, axis=0)
            yield (processed_x_batch, processed_y_batch)

def main(trainingPath, dbPath):
    trainingURLs = listTrainingURLs(trainingPath)
    trainingOutputs = lookupTrainingOutputs(trainingURLs, dbPath)
    model = createTransferModel(len(trainingOutputs[0]))
    train(trainingURLs, trainingOutputs, model)



print("------------------------")
DB_PATH = None
TRAINING_PATH = None

for arg in sys.argv[1:]:
    ss = arg.split("=")
    if len(ss) >= 2:
        if ss[0] == "DB":
            DB_PATH = ss[1]
        elif ss[0] == "TRAINING":
            TRAINING_PATH = ss[1]
        else:
            raise Exception("Unknown Command: " + arg);
    else:
        raise Exception("Unknown Command: " + arg);

if len(sys.argv) == 1 or "--help" in sys.argv:
    print("Usage: ")
    print("python " + sys.argv[0] + " DB=<dbPath> TRAINING=<trainingPath>")
    print("")
    print("Ex: python " + sys.argv[0] + " DB=./images.db TRAINING=./training")
    print("")
else:
    print("DB:", DB_PATH)
    print("TRAINING:", TRAINING_PATH)

if DB_PATH is not None and TRAINING_PATH is not None:
    main(TRAINING_PATH, DB_PATH)
