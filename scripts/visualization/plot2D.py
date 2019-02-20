import numpy as np
import csv;
import sys;
from matplotlib.widgets import CheckButtons

#import Tkinter, tkFileDialog

#root = Tkinter.Tk()
#root.withdraw()
#root.update() #Omg it's an event driven UI in a scripting language... whaaat?
INPUT_FILE = "output-activation_28-pca-tsne.csv" #tkFileDialog.askopenfilename()
COLOR_COL = "compartment"
#root.update() #Omg it's an event driven UI in a scripting language... whaaat?

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

colorBy = []
colorSet = set([])
onlyData = []
for row in data:
    onlyData.append(row[dataStartCol:])
    colorBy.append(row[colorIndex])
    colorSet.add(row[colorIndex])
allColors = sorted(list(colorSet))
colorBy = np.array(colorBy)

from matplotlib import pyplot as plt
target_ids = range(len(colorBy))
#only supports up to 50 colors, change the 10 if you want more.
markers = (['o','v','^','<','>'] * 10)[:len(allColors)]
colors = (['b','g','r','c','m','y','k'] * 10)[:len(allColors)]


#fig= plt.figure(figsize=(4,1.5))
#ax = plt.axes([0.4, 0.2, 0.4, 0.6])
#ax.plot([2,3,1])
#col = (0,0.3,0.75,0.2)
#rax = plt.axes([0.1, 0.2, 0.2, 0.6], facecolor=col )
#check = CheckButtons(rax, ('red', 'blue', 'green'), (1,0,1))
#for r in check.rectangles:
#    r.set_facecolor("blue")
#    r.set_edgecolor("k")
#    r.set_alpha(0.2)
#[ll.set_color("white") for l in check.lines for ll in l]
#[ll.set_linewidth(3) for l in check.lines for ll in l]
#for i, c in enumerate(["r", "b", "g"]):
#    check.labels[i].set_color(c)
#    check.labels[i].set_alpha(0.7)
#plt.show()

fig = plt.figure(figsize=(6, 5))
ax = plt.axes([.3,.1,.6,.8])
scatters = []
toPlot = np.array(onlyData)
print("Time To Plot!")
for i,c,m in zip(target_ids, colors, markers):
    scatter = ax.scatter(
        toPlot[colorBy == allColors[i], 0],
        toPlot[colorBy == allColors[i], 1],
        c=c,
        marker=m,
        visible=False)
    scatters.append(scatter)

rax = plt.axes([0, 0, 0.2, 1])
checkButtons = CheckButtons(rax, allColors, [False] * len(allColors))

def onCheckboxChanged(label):
    for i in range(len(allColors)):
        if label == allColors[i]:
            scatters[i].set_visible(not scatters[i].get_visible())
    plt.draw()

checkButtons.on_clicked(onCheckboxChanged)

print("Showing...")
plt.show()
