import keras
import numpy as np
import os;
import sqlite3;
import json;
import csv;
import sys;

from urlparse import urlparse
from collections import defaultdict
from keras.applications import resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def proteinRowIter(dbConn, proteinListPath):
    if proteinListPath is None:
        cur = dbConn.cursor()
        cur.execute("SELECT url, protein, antibody, cell_line, location FROM image")
        for row in cur:
            yield row
    else:
        proteinList = []
        with open(proteinListPath) as proteinFile:
            for line in proteinFile:
                proteinList.append(line.strip())
    
        print("Proteins: ", len(proteinList))
    
        for protein in proteinList:
            cur = dbConn.cursor()
            cur.execute("SELECT url, protein, antibody, cell_line, location FROM image WHERE protein=?", (protein,))
            for row in cur:
                yield row

def getCachedFileLocation(imageURL, outputDirPath):
    parsedTuple = urlparse(imageURL);
    relativePath = parsedTuple[2][1:];
    outputFile = os.path.join(outputDirPath, relativePath);
    print("Output File: " + outputFile)
    return outputFile

#TODO FIXME HACK: We could do this in batches
#TODO FIXME HACK:  NOTE THAT TO REPLACE THIS WITH A DIFFERENT MODEL, YOU MAY NEED TO EXTRACT THE INPUT PREPROCESSING.
def predict(imageFile, resnet_model):
    # load an image in PIL format
    original = None
    try:
        original = load_img(imageFile, target_size=(224, 224))
    except IOError:
        print("Failed to load file: " + imageFile)
        return None;
    
    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)

    # prepare the image for the ResNet processing
    processed_image = resnet50.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = resnet_model.predict(processed_image)
    return predictions

def genCSVHeader():
    #Grab the output vector names for imagenet
    with open("imagenet_class_index.json") as f:
        classifications = json.load(f);
    
    csvHeaders = ["url", "protein", "antibody", "cell_line", "compartment"];
    for i in range(1000):
        csvHeaders.append(classifications[str(i)][1])
    return csvHeaders

def main(dbPath, proteinListPath, outputDirPath):

    #Load resnet model
    resnet_model = resnet50.ResNet50(weights="imagenet")  #Pre trained on ImageNet
        
    dbConn = None
    try:
        #open the sql connection (input)
        dbConn = sqlite3.connect(dbPath)
        #open the csv file (output)
        with open("output.csv", "w") as csvfile:
            csvHeader = genCSVHeader()
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(csvHeader)

            #For each row queried from sql,
            #Load the image, predict with the model, write the output to the csv.
            for row in proteinRowIter(dbConn, proteinListPath):
                url = row[0]
                filePath = getCachedFileLocation(url, outputDirPath)
                predictions = predict(filePath, resnet_model)
                if predictions is None:
                    continue
                outputRow = []
                flatPreds = np.reshape(predictions,1000)
                for val in row:
                    outputRow.append(val)
                for val in flatPreds:
                    outputRow.append(val)
                writer.writerow(outputRow)
    finally:
        dbConn.close()

print("------------------------")
DB_PATH = "./images.db"
PROTEIN_LIST_PATH = None
IMAGERY_PATH = "./output"

for arg in sys.argv[1:]:
    ss = arg.split("=")
    if len(ss) >= 2:
        if ss[0] == "DB":
            DB_PATH = ss[1]
        elif ss[0] == "PROTEINS":
            PROTEIN_LIST_PATH = ss[1]
        elif ss[0] == "IMAGES":
            IMAGERY_PATH = ss[1]
        else:
            raise Exception("Unknown Command: " + arg);
    else:
        raise Exception("Unknown Command: " + arg);

if len(sys.argv) == 1 or "--help" in sys.argv:
    print("Usage: ")
    print("python " + sys.argv[0] + " DB=<dbPath> PROTEINS=<proteinsPath> IMAGES=<imageryPath>")
    print("")
    print("Ex: python " + sys.argv[0] + " DB=./images.db PROTEINS=./yueProteins.txt IMAGES=./output")
    print("")
else:
    print("DB:", DB_PATH)
    print("PROTEINS:", PROTEIN_LIST_PATH)
    print("IMAGERY:", IMAGERY_PATH)

    main(DB_PATH, PROTEIN_LIST_PATH, IMAGERY_PATH)
