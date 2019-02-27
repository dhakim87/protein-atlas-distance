import numpy as np
import csv;
import sys;
import os;
import random
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import requests

INPUT_FILE = "output-activation_28-merged.csv"
MARKER_COL = "compartment"
COLOR_COL = "KMeans"

class CollectionWrapper:
    def __init__(self, collection):
        self.collection = collection
        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = .3
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)

    def disconnect(self):
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)

#See https://matplotlib.org/gallery/widgets/lasso_selector_demo_sgskip.html
class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    """

    def __init__(self, ax, collections):
        self.canvas = ax.figure.canvas
        self.collections = collections
        self.wrappers = list(CollectionWrapper(x) for x in collections)
        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        for w in self.wrappers:
            w.onselect(verts)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        for w in self.wrappers:
            w.disconnect()
        self.canvas.draw_idle()

def createIndexList(boolList):
    indexList = []
    for i in range(len(boolList)):
        if boolList[i]:
            indexList.append(i)
    return np.array(indexList)

#import Tkinter, tkFileDialog

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

colorIndex = header.index(COLOR_COL)
markerIndex = header.index(MARKER_COL)

colorBy = []
colorSet = set([])
markerBy = []
markerSet = set([])

onlyData = []
for row in data:
    onlyData.append(row[dataStartCol:])
    colorBy.append(row[colorIndex])
    colorSet.add(row[colorIndex])
    markerBy.append(row[markerIndex])
    markerSet.add(row[markerIndex])

allColors = sorted(list(colorSet))
colorBy = np.array(colorBy)
allMarkers = sorted(list(markerSet))
markerBy = np.array(markerBy)

from matplotlib import pyplot as plt
target_ids = range(len(colorBy))

#only supports up to 50 colors, change the 10 if you want more.
#colors = (['b','g','r','c','m','y','k'] * 10)[:len(allColors)]
colors = []
for i in range(50):
    colors.append((random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)))
markers = (['o','v','^','<','>'] * 10)[:len(allColors)]

fig = plt.figure(figsize=(6, 5))
ax = plt.axes([.3,.1,.6,.8])
scatters = []
toPlot = np.array(onlyData)
print("Time To Plot!")
for i,c,m in zip(target_ids, colors, markers):
    pointColors = colorBy[markerBy == allMarkers[i]]
    colorList = []
    for pc in pointColors:
        colorList.append(colors[allColors.index(pc)])
    scatter = ax.scatter(
        toPlot[markerBy == allMarkers[i], 0],
        toPlot[markerBy == allMarkers[i], 1],
        c=colorList,
        marker=m,
        visible=False
    )
    scatters.append(scatter)

rax = plt.axes([0, 0, 0.2, 1])
checkButtons = CheckButtons(rax, allMarkers, [False] * len(allMarkers))
selector = SelectFromCollection(ax, scatters)

def onCheckboxChanged(label):
    for i in range(len(allMarkers)):
        if label == allMarkers[i]:
            scatters[i].set_visible(not scatters[i].get_visible())
    plt.draw()

def accept(event):
    if event.key == "enter":
        print("Visible Scatters:")
        for i in range(len(scatters)):
            if scatters[i].get_visible():
                print(str(i) + ": " + str(allMarkers[i]))
                print("Selected points:")
                wrapper = selector.wrappers[i]
                rawIndices = createIndexList(markerBy == allMarkers[i])
                print(wrapper.xys[wrapper.ind])
                print(rawIndices[wrapper.ind])
                for rawIndex in rawIndices[wrapper.ind]:
                    print data[rawIndex][0:5]
    if event.key == "a":
        print("Downloading 10 random images")
        for i in range(len(scatters)):
            if scatters[i].get_visible():
                wrapper = selector.wrappers[i]
                rawIndices = createIndexList(markerBy == allMarkers[i])
                for rawIndex in rawIndices[wrapper.ind]:
                    print data[rawIndex][0:5]
                rawShuffle = list(rawIndices[wrapper.ind])
                random.shuffle(rawShuffle)
                for rawIndex in rawShuffle[:min(10,len(rawShuffle))]:
                    url = data[rawIndex][0]
                    print(url)
                    if not os.path.exists("A"):
                        os.makedirs("A")
                    with open("A/"+str(rawIndex)+".jpg", 'w') as f:
                        resp = requests.get(url)
                        f.write(resp.content)
    if event.key == "b":
        print("Downloading 10 random images")
        for i in range(len(scatters)):
            if scatters[i].get_visible():
                wrapper = selector.wrappers[i]
                rawIndices = createIndexList(markerBy == allMarkers[i])
                for rawIndex in rawIndices[wrapper.ind]:
                    print data[rawIndex][0:5]
                rawShuffle = list(rawIndices[wrapper.ind])
                random.shuffle(rawShuffle)
                for rawIndex in rawShuffle[:min(10,len(rawShuffle))]:
                    url = data[rawIndex][0]
                    print(url)
                    if not os.path.exists("B"):
                        os.makedirs("B")
                    with open("B/"+str(rawIndex)+".jpg", 'w') as f:
                        resp = requests.get(url)
                        f.write(resp.content)

fig.canvas.mpl_connect("key_press_event", accept)

checkButtons.on_clicked(onCheckboxChanged)

print("Showing...")
plt.show()
