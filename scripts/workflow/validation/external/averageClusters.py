import csv;
import sys;
import math;
import numpy as np
from collections import defaultdict
import random;

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

def filterData(data, filterColIndex, filterVal):
    filteredData = []
    for row in data:
        if row[filterColIndex] == filterVal:
            filteredData.append(row)
    return filteredData

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

def findClusterCenter(rows, startColIndex):
    clusterSums = [0] * (len(rows[0]) - startColIndex)
    for row in rows:
        for col in range(startColIndex, len(rows[0])):
            clusterSums[col-startColIndex] += row[col]
    for i in range(len(clusterSums)):
        clusterSums[i] /= len(rows)
    return clusterSums

def writeClusterToClusterDist(clusterCenters, csvWriter):
    num = 0
    for A in clusterCenters:
        for B in clusterCenters:
            if A > B:
                continue
            num += 1
            if num % 1000 == 0:
                print num, "/", (len(clusterCenters) * (len(clusterCenters)+1) / 2)
            sum = 0
            for col in range(len(clusterCenters[A])):
                a = clusterCenters[A][col]
                b = clusterCenters[B][col]
                sum += (b-a)*(b-a)
            dist = math.sqrt(sum)
            csvWriter.writerow([A, B, dist])

def main(clusterColumn, inputFile, filterCol, filterVal, outputFile):
    #START
    #Config
    #Read the input file (Maybe make user choose a file or take on cmd line?)
    (header, data) = readCSV(inputFile)
    
    #File is assumed to be metadata headers followed by data headers, so we just find the first parsable float in the first row and mark that as the start of our data
    dataStartCol = findDataStart(data[0])
    
    filterColIndex = header.index(filterCol)
    print("Filter Index: ", filterColIndex)
    data = filterData(data, filterColIndex, filterVal)

    #Let user check that we parsed it correctly
    print "Metadata Headers"
    print header[0:dataStartCol]
    print "Data Column Count: ", len(data[0])-dataStartCol

    #Convert all the floating point data to floats
    castData(data, dataStartCol)

    clusteredData = defineClusters(header, data, clusterColumn)

    clusterCenters = {}
    for cluster in clusteredData:
        center = findClusterCenter(clusteredData[cluster], dataStartCol)
        clusterCenters[cluster] = center

    with open(outputFile, "w") as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        outHeader = [clusterColumn] + header[dataStartCol:]
        writer.writerow(outHeader)
        
        for cluster in clusterCenters:
            outRow = [cluster] + clusterCenters[cluster]
            writer.writerow(outRow)

    print("Computing Pairwise Distances...")


    with open(outputFile[:-4] + "_dists.csv", "w") as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        outHeader = [clusterColumn + 'A', clusterColumn + 'B', "Distance"]
        writer.writerow(outHeader)
        writeClusterToClusterDist(clusterCenters, writer)

print("------------------------")
CLUSTER_COLUMN = "compartment"
FILTER_COLUMN = None
FILTER_VALUE = None
INPUT_PATH = None
OUTPUT_PATH = None

for arg in sys.argv[1:]:
    ss = arg.split("=")
    if len(ss) >= 2:
        if ss[0] == "CLUSTER":
            CLUSTER_COLUMN = ss[1]
        elif ss[0] == "INPUT":
            INPUT_PATH = ss[1]
        elif ss[0] == "OUTPUT":
            OUTPUT_PATH = ss[1]
        elif ss[0] == "FILTER_COL":
            FILTER_COLUMN = ss[1]
        elif ss[0] == "FILTER_VAL":
            FILTER_VALUE = ss[1]
        else:
            raise Exception("Unknown Command: " + arg);
    else:
        raise Exception("Unknown Command: " + arg);

if len(sys.argv) == 1 or "--help" in sys.argv:
    print "Usage: "
    print "python " + sys.argv[0] + " CLUSTER=<columnToClusterOn> FILTER_COL=<columnToFilterOn> FILTER_VAL=<ValueToFilterFor> INPUT=<pathToCSVFile> OUTPUT=<pathToNewOutputFile>"
    print ""
    print "Ex: python " + sys.argv[0] + " CLUSTER=compartment FILTER_COL=cell_line FILTER_VAL=\"U-2 OS\" INPUT=./yueProteins.csv OUTPUT=clusteredProteins.csv"
    print ""

print "CLUSTER_COLUMN:", CLUSTER_COLUMN
print "FILTER_COLUMN: ", FILTER_COLUMN
print "FILTER_VALUE: ", FILTER_VALUE
print "INPUT_PATH:", INPUT_PATH
print "OUTPUT_PATH:", OUTPUT_PATH

#Sick of screwing up my scoping, so I made a main method!
if CLUSTER_COLUMN is not None and INPUT_PATH is not None:
    main(CLUSTER_COLUMN, INPUT_PATH, FILTER_COLUMN, FILTER_VALUE, OUTPUT_PATH)
