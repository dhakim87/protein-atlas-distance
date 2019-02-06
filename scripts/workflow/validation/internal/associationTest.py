from collections import defaultdict
import csv
import seaborn as sns
import matplotlib.pylab as plt

counter = defaultdict(int)
compartmentSet = set([])
kmeansSet = set([])

header = None
with open("clusterToLabels.csv") as inputFile:
    reader = csv.reader(inputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        if header is None:
            header = row
            continue
        #row[4],row[5] is compartment,kMeans
        compartmentSet.add(row[4])
        kmeansSet.add(int(row[5]))
        counter[(row[4], row[5])] += 1

compartmentList = sorted(list(compartmentSet))
kmeansList = sorted(list(kmeansSet))

xLabels = kmeansList
yLabels = compartmentList

H = []
for y in compartmentList:
    row = []
    for x in kmeansList:
        row.append(counter[(y, str(x))])
    H.append(row)

ax = sns.heatmap(H, linewidth=0.5, xticklabels=xLabels, yticklabels=yLabels, square=True)
plt.show()

