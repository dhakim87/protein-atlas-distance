import csv;
import sys;
import math;
import numpy as np
from collections import defaultdict
import random;
from sklearn.cluster import KMeans

def readCSV(filename, parseHeader = True):
    header = None
    data = []
    with open(filename) as csvFile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if parseHeader:
                header = row
                parseHeader = False
            else:
                data.append(row)
    return (header, data)

def findDataStart(nonHeaderRow):
    for index in range(len(nonHeaderRow)):
        try:
            float(nonHeaderRow[index])
            return index
        except ValueError:
            continue

def castData(data, dataStartCol):
    for row in data:
        for col in range(dataStartCol, len(row)):
            row[col] = float(row[col])

def pickKmeansClusters(header, data, numberOfClusters, dataStartCol):
    dataCopy = list(data)
    for i in range(len(dataCopy)):
        dataCopy[i] = np.array(dataCopy[i][dataStartCol:])
    dataCopy = np.stack(dataCopy)
    print dataCopy.shape
    kmeans = KMeans(numberOfClusters).fit(dataCopy)

    clusterCenters = kmeans.cluster_centers_
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    n_iter = kmeans.n_iter_

    print "Cluster Centers", clusterCenters
    print "Labels", labels
    print "Inertia", inertia
    print "Iterations", n_iter

    kmeansClusteredData = defaultdict(list)
    for i in range(len(dataCopy)):
        kmeansClusteredData[labels[i]].append(data[i])
    return kmeansClusteredData

def main(csvFile, outputFileName, k):
    #START
    #Config

    #Read the input file (Maybe make user choose a file or take on cmd line?)
    (header, data) = readCSV(csvFile)

    #File is assumed to be metadata headers followed by data headers, so we just find the first parsable float in the first row and mark that as the start of our data
    dataStartCol = findDataStart(data[0])


    #Let user check that we parsed it correctly
    print "Metadata Headers"
    print header[0:dataStartCol]
    print "Data Column Count: ", len(data[0])-dataStartCol

    #Convert all the floating point data to floats
    castData(data, dataStartCol)
    
    print("Clustering with k means")
    #And we will compare it to kmeans clustered data with the same number of clusters
    kmeansClusteredData = pickKmeansClusters(header, data, k, dataStartCol)

    print("Printing clusterToLabels.csv")
    with open(outputFileName, "w") as outputFile:
        writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header[0:dataStartCol] + ["KMeans"])
        for label in kmeansClusteredData:
            for dataRow in kmeansClusteredData[label]:
                labelStr = str(label)
                if labelStr is None or len(labelStr) == 0:
                    labelStr = "None"
                writer.writerow(dataRow[0:dataStartCol] + [labelStr])

print("------------------------")
INPUT_CSV = None
OUTPUT_CSV = None
K = None

for arg in sys.argv[1:]:
    ss = arg.split("=")
    if len(ss) >= 2:
        if ss[0] == "INPUT":
            INPUT_CSV = ss[1]
        elif ss[0] == "OUTPUT":
            OUTPUT_CSV = ss[1]
        elif ss[0] == "K":
            K = int(ss[1])
        else:
            raise Exception("Unknown Command: " + arg);
    else:
        raise Exception("Unknown Command: " + arg);

if len(sys.argv) == 1 or "--help" in sys.argv:
    print "Usage: "
    print "python " + sys.argv[0] + " INPUT=<pathToCSVFile> OUTPUT=<pathToCSVFile> K=<Num Clusters>"
    print ""
    print "Ex: python " + sys.argv[0] + " INPUT=./predictions.csv OUTPUT=./kmeansClusters.csv K=8"
    print ""

print "INPUT_PATH:", INPUT_CSV
print "OUTPUT_PATH:", OUTPUT_CSV
print "K:", K

#Sick of screwing up my scoping, so I made a main method!
if INPUT_CSV is not None and OUTPUT_CSV is not None and K is not None:
    main(INPUT_CSV, OUTPUT_CSV, K)
