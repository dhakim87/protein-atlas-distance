import numpy as np
import csv;
import sys;

from sklearn.decomposition import PCA
from matplotlib.widgets import CheckButtons

NUM_PCA_COMPONENTS = 30
INPUT_FILE = "output-activation_28.csv"

dotIndex = INPUT_FILE.rindex('.')
OUTPUT_FILE = INPUT_FILE[0:dotIndex] + "-pca.csv"

print INPUT_FILE + " -> " + OUTPUT_FILE

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

#permutation500 = np.random.permutation(500)
#X = X[permutation500]
#Y = Y[permutation500]

print X.shape
print Y.shape
#X = np.array([[-10, -1], [-10, -2], [-10, -3], [10, 1], [10, 2], [10, 3]])
#Y = np.array([0,0,0,1,1,1])

#PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
#  svd_solver='auto', tol=0.0, whiten=False)
pca = PCA(n_components=NUM_PCA_COMPONENTS)
pca.fit(X)

print("X Shape: " + str(X.shape))
print("Vectors: ")
print(pca.components_)
print("Explained Variance: ")
print(pca.explained_variance_ratio_)
print("Singular Values: ")
print(pca.singular_values_)

XPCA = pca.transform(X);
print X.shape
print XPCA.shape

with open(OUTPUT_FILE, 'w') as csvFile:
    writer = csv.writer(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    headerRow = header[0:dataStartCol]
    for i in range(NUM_PCA_COMPONENTS):
        headerRow.append("PCA"+str(i+1))
    writer.writerow(headerRow)
    for row in range(XPCA.shape[0]):
        r = []
        for val in data[row][0:dataStartCol]:
            r.append(val)
        for val in XPCA[row]:
            r.append(val)
        writer.writerow(r)
