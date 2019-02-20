import numpy as np
import csv;
import sys;

from sklearn.manifold import TSNE
from matplotlib.widgets import CheckButtons

print("------------------------")
INPUT_FILE = None

for arg in sys.argv[1:]:
    ss = arg.split("=")
    if len(ss) >= 2:
        if ss[0] == "INPUT":
            INPUT_FILE = ss[1]
        else:
            raise Exception("Unknown Command: " + arg);
    else:
        raise Exception("Unknown Command: " + arg);

if len(sys.argv) == 1 or "--help" in sys.argv:
    print "Usage: "
    print "python " + sys.argv[0] + " INPUT=<pathToCSVFile>"
    print ""
    print "Ex: python " + sys.argv[0] + " INPUT=output-activation_28.csv"

if INPUT_FILE == None:
    print("------------------------")
    sys.exit()

dotIndex = INPUT_FILE.rindex('.')
OUTPUT_FILE = INPUT_FILE[0:dotIndex] + "-tsne.csv"

print INPUT_FILE + " -> " + OUTPUT_FILE
print("------------------------")

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

#Read the input file (Maybe make user choose a file or take on cmd line?)
(header, data) = readCSV(INPUT_FILE)

#File is assumed to be metadata headers followed by data headers, so we just find the first parsable float in the first row and mark that as the start of our data
dataStartCol = findDataStart(data[0])

#Let user check that we parsed it correctly
print "Metadata Headers"
print header[0:dataStartCol]
print "Data Column Count: ", len(data[0])-dataStartCol

#Convert all the floating point data to floats
castData(data, dataStartCol)

onlyData = []
compartment = []
compartmentSet = set([])
for row in data:
    onlyData.append(row[dataStartCol:])
    compartment.append(row[4])
    compartmentSet.add(row[4])

X = np.stack(onlyData)
Y = np.array(compartment)

print X.shape
print Y.shape

print "Starting TSNE"
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

print X_2d
print X_2d.shape

with open(OUTPUT_FILE, 'w') as csvFile:
    writer = csv.writer(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header[0:dataStartCol] + ["TSNE1", "TSNE2"])
    for row in range(X_2d.shape[0]):
        r = []
        for val in data[row][0:dataStartCol]:
            r.append(val)
        for val in X_2d[row]:
            r.append(val)
        writer.writerow(r)
