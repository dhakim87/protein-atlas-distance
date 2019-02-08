import csv;
import sys;
import math;
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random;
from scipy.stats import ks_2samp
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
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

def defineClusters(header, data, columnName):
    clusterBy = columnName
    clusterByIndex = header.index(clusterBy)
    clusteredData = defaultdict(list)
    for row in data:
        clusteredData[row[clusterByIndex]].append(row)
    return clusteredData

def pickRandomClusters(header, data, numberOfClusters):
    dataCopy = list(data)
    random.shuffle(dataCopy)
    
    randomlyClusteredData = defaultdict(list)
    for i in range(len(dataCopy)):
        randomlyClusteredData[i%numberOfClusters].append(dataCopy[i])
    return randomlyClusteredData

def pickRandomSameSizeClusters(header, data, realClusters):
    dataCopy = list(data)
    random.shuffle(dataCopy)

    randomlyClusteredData = defaultdict(list)
    randomRowIndex = 0;
    for key in realClusters:
        for val in realClusters[key]:
            randomlyClusteredData[key].append(dataCopy[randomRowIndex])
            randomRowIndex += 1
    return randomlyClusteredData

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

def computeAllPointToPointDists(rows, startColIndex):
    dists = []
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            sum = 0
            for col in range(startColIndex, len(rows[0])):
                a = rows[i][col]
                b = rows[j][col]
                sum += (b-a)*(b-a)
            dist = math.sqrt(sum)
            dists.append(dist)
    return dists

def computeMaxPointToPointDist(rows, startColIndex):
    maxDist = 0;
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            sum = 0
            for col in range(startColIndex, len(rows[0])):
                a = rows[i][col]
                b = rows[j][col]
                sum += (b-a)*(b-a)
            dist = math.sqrt(sum)
            maxDist = max(dist, maxDist)
    return maxDist

def computeAABB(rows, startColIndex):
    numCols = len(rows[0]) - startColIndex
    mins = []
    maxs = []
    for col in range(startColIndex, len(rows[0])):
        mins.append(rows[0][col])
        maxs.append(rows[0][col])
    
    for i in range(len(rows)):
        for col in range(startColIndex, len(rows[0])):
            mins[col-startColIndex] = min(mins[col-startColIndex], rows[i][col])
            maxs[col-startColIndex] = max(maxs[col-startColIndex], rows[i][col])

    return (mins,maxs)

def runSilhouetteAnalysis(clusteredData, startColIndex, plotTitle):
    X = []
    Labels = []
    for key in clusteredData:
        for row in clusteredData[key]:
            X.append(row[startColIndex:])
            Labels.append(key)

    score = silhouette_score(X, Labels)
    print "Score: ", score
    silSamples = silhouette_samples(X, Labels)
#    print "Samples: ", silSamples

    print("Constructed silhouette plot...")
    plt.bar(
        np.arange(len(silSamples)),
        silSamples,
        align='center',
        )

    plt.title(plotTitle)
    plt.show()

def main(clusterColumn, csvFile):
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

    print("Clustering on " + clusterColumn)
    #Cluster the data (Would be nice to support concatenating keys, but one col should suffice for now.
    clusteredData = defineClusters(header, data, clusterColumn)

    print("Clustering with randomized uniform size clusters")
    #We will compare this to randomly clustered data with the same number of clusters (where all clusters are the same size)
    randomlyClusteredData = pickRandomClusters(header, data, len(clusteredData))

    print("Clustering with randomized match size clusters")
    #And we will compare it to randomly clustered data with the same number of clusters where cluster size matches the real clusters
    randomlyClusteredData2 = pickRandomSameSizeClusters(header, data, clusteredData)
    
    print("Clustering with k means")
    #And we will compare it to kmeans clustered data with the same number of clusters
    kmeansClusteredData = pickKmeansClusters(header, data, len(clusteredData), dataStartCol)

    print("Printing clusterToLabels.csv")
    with open("clusterToLabelsKMeans2.csv", "w") as outputFile:
        writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header[0:dataStartCol] + ["KMeans"])
        for label in kmeansClusteredData:
            for dataRow in kmeansClusteredData[label]:
                labelStr = str(label)
                if labelStr is None or len(labelStr) == 0:
                    labelStr = "None"
                writer.writerow(dataRow[0:dataStartCol] + [labelStr])

    print("Running silhouette analysis")
    #Generate silhouette scores on the three clusterings, real, randomUniformSize and randomRealSize
    runSilhouetteAnalysis(clusteredData, dataStartCol, 'Cluster By Metadata Silhouette Score ' + clusterColumn)
    runSilhouetteAnalysis(randomlyClusteredData, dataStartCol, 'Random Uniform Size Cluster Silhouette Score ' + clusterColumn)
    runSilhouetteAnalysis(randomlyClusteredData2, dataStartCol, 'Random Matching Sized Silhouette Score ' + clusterColumn)
    runSilhouetteAnalysis(kmeansClusteredData, dataStartCol, 'Random Matching Sized Silhouette Score ' + clusterColumn)

    #Compute aggregate characteristics over each cluster
    clusteredPointToPointDist = {}
    for clusterKey in clusteredData:
        clusteredPointToPointDist[clusterKey] = computeMaxPointToPointDist(clusteredData[clusterKey], dataStartCol)

    #Compute aggregate characteristics over completely RANDOM clusters for comparison
    randomlyClusteredPointToPointDist = {}
    for clusterKey in randomlyClusteredData:
        randomlyClusteredPointToPointDist[clusterKey] = computeMaxPointToPointDist(randomlyClusteredData[clusterKey], dataStartCol)

    #Compute aggregate characteristics over RANDOM clusters with the same structure for comparison
    randomlyClusteredPointToPointDist2 = {}
    for clusterKey in randomlyClusteredData2:
        randomlyClusteredPointToPointDist2[clusterKey] = computeMaxPointToPointDist(randomlyClusteredData2[clusterKey], dataStartCol)

    print "Number of " + clusterColumn + "s:", len(clusteredPointToPointDist)
    isItRandomWithUniformClusterSize = ks_2samp(clusteredPointToPointDist.values(), randomlyClusteredPointToPointDist.values())
    isItRandomWithChosenClusterSize = ks_2samp(clusteredPointToPointDist.values(), randomlyClusteredPointToPointDist2.values())

    print "---------------------------------------------------"
    print "Are the clusters real? "
    print "\t KS Test " + clusterColumn + " vs Random Uniform Size", isItRandomWithUniformClusterSize
    print "\t KS Test " + clusterColumn + " vs Random Matching Cluster Size", isItRandomWithChosenClusterSize
    print "---------------------------------------------------"

    plt.bar(
        np.arange(len(clusteredPointToPointDist))-.2, #X
        sorted(clusteredPointToPointDist.values()), #Y
        align='center',
        alpha=0.5,
        color='b',
        width=.2)
    plt.bar(
        np.arange(len(randomlyClusteredPointToPointDist))+.0, #X
        sorted(randomlyClusteredPointToPointDist.values()), #Y
        align='center',
        alpha=0.5,
        color='g',
        width=.2)
    plt.bar(
        np.arange(len(randomlyClusteredPointToPointDist2))+.2, #X
        sorted(randomlyClusteredPointToPointDist2.values()), #Y
        align='center',
        alpha=0.5,
        color='r',
        width=.2)

    plt.title('P2P Cluster Max Dist ' + clusterColumn + ' vs Random')
    plt.show()

print("------------------------")
CLUSTER_COLUMN = "compartment"
INPUT_CSV = None

for arg in sys.argv[1:]:
    ss = arg.split("=")
    if len(ss) >= 2:
        if ss[0] == "COLUMN":
            CLUSTER_COLUMN = ss[1]
        elif ss[0] == "INPUT":
            INPUT_CSV = ss[1]
        else:
            raise Exception("Unknown Command: " + arg);
    else:
        raise Exception("Unknown Command: " + arg);

if len(sys.argv) == 1 or "--help" in sys.argv:
    print "Usage: "
    print "python " + sys.argv[0] + " COLUMN=<columnToClusterOn> INPUT=<pathToCSVFile>"
    print ""
    print "Ex: python " + sys.argv[0] + " COLUMN=compartment INPUT=./output.csv"
    print ""

print "CLUSTER_COLUMN:", CLUSTER_COLUMN
print "CSV_PATH:", INPUT_CSV

#Sick of screwing up my scoping, so I made a main method!
if CLUSTER_COLUMN is not None and INPUT_CSV is not None:
    main(CLUSTER_COLUMN, INPUT_CSV)
