import csv;
import sys;
import math;

#Update these with your target file paths: (and deal with the fact that the header still says YueDist and DanDist in it.)
DAN_DIST_FILE = "u2osProteinCenters_dists.csv"
YUE_DIST_FILE = "u2os_similarity_yue.tsv"
COMBINED_OUTPUT_FILE = "u2os_similarity_combined.csv"
ERROR_MISSING_FILE = "u2os_missingProteins.csv"



#Read my csv
danDistanceDict = {}
numRowsRead = 0;
with open(DAN_DIST_FILE) as csvFile:
    print "Reading In Dan Distances"
    reader = csv.reader(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    skipHeader = True
    for row in reader:
        numRowsRead += 1
        if numRowsRead % 1000 == 0:
            print numRowsRead
        if skipHeader:
            skipHeader = False
            continue
        else:
            first = row[0]
            second = row[1]
            if first > second:
                first = row[1]
                second = row[0]
            danDistanceDict[(first, second)] = float(row[2])

yueDistanceDict = {}
numRowsRead = 0
with open(YUE_DIST_FILE) as csvFile:
    print "Reading In Yue Distances"
    reader = csv.reader(csvFile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    skipHeader = False
    for row in reader:
        numRowsRead += 1
        if numRowsRead % 1000 == 0:
            print numRowsRead
        if skipHeader:
            skipHeader = False
            continue
        else:
            first = row[0]
            second = row[1]
            if first > second:
                first = row[1]
                second = row[0]
            yueDistanceDict[(first, second)] = float(row[2])

with open(COMBINED_OUTPUT_FILE, "w") as csvFile:
    with open(ERROR_MISSING_FILE, "w") as csvErrors:
        writer = csv.writer(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(("proteinA","proteinB","DanDist","YueDist"))
        errors = csv.writer(csvErrors, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        errors.writerow(("pairA","pairB", "OnlyIn"))
        
        for pair in danDistanceDict:
            if pair in yueDistanceDict:
                writer.writerow((pair[0],pair[1],danDistanceDict[pair],yueDistanceDict[pair]))
            else:
                errors.writerow((pair[0],pair[1], "DAN"))
        for pair in yueDistanceDict:
            if pair not in danDistanceDict:
                errors.writerow((pair[0],pair[1], "YUE"))

