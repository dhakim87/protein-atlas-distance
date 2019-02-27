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
from keras.utils import Sequence
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import random
import sqlite3
import uuid

LAST_FROZEN_LAYER = "add_4"
CHOP_LAYER = "activation_28"

NUM_EPOCHS = 150
NUM_OUTPUT_CATEGORIES = 32
SHUFFLE_SEED = 127
TRAINING_FRACTION = .75
CATEGORY_TO_NUMBER = {}

class AutoSave(Callback):
    def __init__(self, model, trainingInstanceName):
        super(AutoSave, self).__init__()
        self.model = model
        self.trainingInstanceName = trainingInstanceName
    
    def on_epoch_end(self, epoch, logs=None):
        print("On Epoch End: " + str(epoch))
        if epoch % 5 == 0:
            print("Autosave...")
            self.model.save(self.trainingInstanceName + "-E" + str(epoch) + ".h5")

#Create a generator object to feed to Keras so that we can use its built in fit_generator functionality
#See https://medium.com/datadriveninvestor/keras-training-on-large-datasets-3e9d9dbc09d4
class BatchGenerator(Sequence):
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.images) / float(self.batch_size))

    def __getitem__(self, idx):
        #Does this index out of bounds?
        x_batch = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return processImageBatch(x_batch), processCategories(y_batch)

def createTransferModel():
    resnetModel = resnet50.ResNet50(weights="imagenet", include_top=False)  #Pre trained on ImageNet but without the final layer reducing to 1000 categories
    
    chop = resnetModel.get_layer(CHOP_LAYER);
    x = chop.output
    x = GlobalAveragePooling2D()(x)
    preds = Dense(NUM_OUTPUT_CATEGORIES, activation='softmax')(x)

    transferModel = Model(inputs=resnetModel.input, outputs=preds)

    #Print the model to a file to check that we modified it as expected
#    plot_model(transferModel, to_file='transfer_model.svg');
#    print transferModel.summary();

    for layer in transferModel.layers:
        layer.trainable = False
        if layer.name == LAST_FROZEN_LAYER:
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
        start = imageURL.index("/images/")
        if start < 0:
            raise Exception("Couldn't parse image URL")
        imgURLTuple = ("http://v18.proteinatlas.org" + imageURL[start:],)
        cur.execute("SELECT * FROM image WHERE url = ?", imgURLTuple)
        rows = cur.fetchall()
        if len(rows) < 1:
            raise Exception("COULDNT FIND ROW FOR: " + imgURLTuple[0])
        if len(rows) > 1:
            print("IGNORING MULTIPLE ROWS FOR URL: " + imgURLTuple[0]);
        row = rows[0]
#        print("URL: " + row[0] + " Protein: " + row[1] + " antibody: " + row[2] + " cell_line: " + row[3] + " location: " + row[4]);
        categories.append(row[4])
    conn.close()

    numberGen = 0
    for category in categories:
        if category not in CATEGORY_TO_NUMBER:
            CATEGORY_TO_NUMBER[category] = numberGen;
            numberGen += 1

    if len(CATEGORY_TO_NUMBER) > NUM_OUTPUT_CATEGORIES:
        print("NOT ENOUGH CATEGORIES IN OUTPUT")

    sortedCategories = [None] * NUM_OUTPUT_CATEGORIES
    for c in CATEGORY_TO_NUMBER:
        sortedCategories[CATEGORY_TO_NUMBER[c]] = c

    for i in range(len(sortedCategories)):
        print(str(sortedCategories[i]) + " -> " + str(i))

    return categories;

def train(imageURLs, categories, model):
    #Each epoch will train over all of our training images in a random order with a particular batch size.  We'll mimic (or directly use, if I can figure out how) the Keras' ImageDataGenerator.

    #See full example at https://keras.io/preprocessing/image/
    x_train = imageURLs;
    y_train = categories;
    for e in range(5):
        model.save("modelPreEpoch"+str(e)+".h5")
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in customFlow(x_train, y_train, 32):
            print("Batch: " + str(batches))
            model.fit(x_batch, y_batch)
            batches += 1
            
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
    model.save("modelFinal.h5")

def trainWithGenerator(imageURLs, categories, model):
    trainingRunIdentifier = uuid.uuid4();
    
    allTuples = []
    for i in range(len(imageURLs)):
        allTuples.append((imageURLs[i], categories[i]))
    random.Random(SHUFFLE_SEED).shuffle(allTuples)

    cutoff = int(len(allTuples) * TRAINING_FRACTION)
    trainingTuples = allTuples[0:cutoff]
    validationTuples = allTuples[cutoff:]

    print("Num Training:   " + str(len(trainingTuples)))
    print("Num Validation: " + str(len(validationTuples)))

    trainingImages = [i[0] for i in trainingTuples]
    trainingCategories = [i[1] for i in trainingTuples]

    validationImages = [i[0] for i in validationTuples]
    validationCategories = [i[1] for i in validationTuples]

    trainingGen = BatchGenerator(trainingImages, trainingCategories, 32)
    validationGen = BatchGenerator(validationImages, validationCategories, 32)

    H = model.fit_generator(generator=trainingGen,
                          steps_per_epoch=len(trainingGen),
                          epochs=NUM_EPOCHS,
                          verbose=1,
                          validation_data=validationGen,
                          validation_steps=len(validationGen),
                          callbacks=[AutoSave(model, str(trainingRunIdentifier))],
                          use_multiprocessing=True,
                          workers=16,
                          max_queue_size=32,
                          shuffle=True)

    print("Training Run Identifier: " + str(trainingRunIdentifier))
    model.save("modelFinal-" + str(trainingRunIdentifier) + ".h5")

    # plot the training loss and accuracy
#    plt.figure()
#    plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
#    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
#    plt.plot(np.arange(0, NUM_EPOCHS), H.history["acc"], label="train_acc")
#    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_acc"], label="val_acc")
#    plt.title("Training Loss and Accuracy")
#    plt.xlabel("Epoch #")
#    plt.ylabel("Loss/Accuracy")
#    plt.legend(loc="upper right")
#    plt.savefig(str(trainingRunIdentifier)+".png")
#    print("History Image Saved To: " + str(trainingRunIdentifier) + ".png")


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
                batchHots = []
    
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

def processCategories(y_batch):
    oneHots = []
    for category in y_batch:
        oneHot = [0] * NUM_OUTPUT_CATEGORIES
        oneHot[CATEGORY_TO_NUMBER[category]] = 1
        oneHots.append(oneHot)
    return np.stack(oneHots, axis=0);

def customFlow(x_train, y_train, batch_size=32):
    while True:
        for x_batch, y_batch in randomBatchGenerator(x_train, y_train, batch_size):
            print(zip(x_batch, y_batch))
            processed_x_batch = processImageBatch(x_batch)
            processed_y_batch = processCategories(y_batch)
            yield (processed_x_batch, processed_y_batch)

def main(trainingPath, dbPath):
    trainingURLs = listTrainingURLs(trainingPath)
    trainingOutputs = lookupTrainingOutputs(trainingURLs, dbPath)
    model = createTransferModel()
    trainWithGenerator(trainingURLs, trainingOutputs, model)

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
