from collections import defaultdict
import csv
import seaborn as sns
import matplotlib.pylab as plt
from scipy.stats import chi2_contingency
import numpy
import matplotlib.patches as mpatches

class Annotater:  #Tater?  Tator?  Mmm.. taters.
    def __init__(self, dataset, ax, fig):
        self.dataset = dataset
        self.ax = ax
        self.fig = fig
    
    def onMotion(self, event):
        if not event.inaxes is self.ax:
            return

        xint = int(event.xdata)
        yint = int(event.ydata)
        hoverRect = mpatches.Rectangle((xint, yint),1,1,fill=False,edgecolor='orange',linewidth=1)
        annotation = str(self.dataset[yint][xint])
        #x+1, y -> One cell up and one cell to the right the way text gets drawn
        tempText = self.ax.text(xint, yint+1, annotation, horizontalalignment='left', size='large', color='white', weight='semibold')
        print "X:", xint, "Y:", yint, "Val:", annotation
        
        self.ax.add_patch(hoverRect)
        self.fig.canvas.draw()
        tempText.remove()
        hoverRect.remove()

counter = defaultdict(int)
compartmentSet = set([])
kmeansSet = set([])

areLabelsIntegers = True #Try parsing labels as integers for nice sorting purposes

header = None
with open("clusterToLabelsKMeans.csv") as inputFile:
    reader = csv.reader(inputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        if header is None:
            header = row
            continue
        #row[4],row[5] is compartment,kMeans
        compartmentSet.add(row[4])
        if areLabelsIntegers:
            try:
                x = int(row[5])
            except:
                areLabelsIntegers = False

        if areLabelsIntegers:
            kmeansSet.add(int(row[5]))
        else:
            kmeansSet.add(row[5])
        counter[(row[4], row[5])] += 1

compartmentList = sorted(list(compartmentSet))
kmeansList = sorted(list(kmeansSet))

xLabels = kmeansList
yLabels = compartmentList

#Create observation matrix
O = []
for y in compartmentList:
    row = []
    for x in kmeansList:
        row.append(counter[(y, str(x))])
    O.append(row)

#Run Chi Sq Contingency
chi2, p, dof, E = chi2_contingency(O)
print "Chi Sq: ", chi2
print "P-Value: ", p
print "Degrees Of Freedom: ", dof

OMinusE = numpy.subtract(O,E)
OMinusESqOverE = numpy.divide(numpy.multiply(OMinusE, OMinusE), E)
print "SUM: ", numpy.sum(OMinusESqOverE)


#PLOT OBSERVED
ax = sns.heatmap(O, linewidth=0.5, xticklabels=xLabels, yticklabels=yLabels, square=True, annot=False)
plt.title("Observed")

annotate = Annotater(O, plt.gca(), plt.gcf())
cid = plt.gcf().canvas.mpl_connect('motion_notify_event', annotate.onMotion)
plt.show()
plt.gcf().canvas.mpl_disconnect(cid)

#PLOT EXPECTED
ax = sns.heatmap(E, linewidth=0.5, xticklabels=xLabels, yticklabels=yLabels, square=True, annot=False)
plt.title("Expected")
annotate = Annotater(E, plt.gca(), plt.gcf())
cid = plt.gcf().canvas.mpl_connect('motion_notify_event', annotate.onMotion)
plt.show()
plt.gcf().canvas.mpl_disconnect(cid)

#PLOT OBSERVED - EXPECTED
ax = sns.heatmap(OMinusE, linewidth=0.5, xticklabels=xLabels, yticklabels=yLabels, square=True, annot=False)
plt.title("Observed - Expected")
annotate = Annotater(OMinusE, plt.gca(), plt.gcf())
cid = plt.gcf().canvas.mpl_connect('motion_notify_event', annotate.onMotion)
plt.show()
plt.gcf().canvas.mpl_disconnect(cid)

#PLOT (OBSERVED - EXPECTED)^2 / EXPECTED
ax = sns.heatmap(OMinusESqOverE, linewidth=0.5, xticklabels=xLabels, yticklabels=yLabels, square=True, annot=False)
plt.title("(O-E)^2 / E")
annotate = Annotater(OMinusESqOverE, plt.gca(), plt.gcf())
cid = plt.gcf().canvas.mpl_connect('motion_notify_event', annotate.onMotion)
plt.show()
plt.gcf().canvas.mpl_disconnect(cid)

#PLOT E AND (O-E)^2/E
plt.subplot(1, 2, 1)
ax = sns.heatmap(E, linewidth=0.5, square=True, annot=False)
ax.set_title("E")
annotateLeft = Annotater(E, ax, plt.gcf())
cidLeft = plt.gcf().canvas.mpl_connect('motion_notify_event', annotateLeft.onMotion)

plt.subplot(1, 2, 2)
ax = sns.heatmap(OMinusESqOverE, linewidth=0.5, square=True, annot=False)
ax.set_title("(O-E)^2 / E")
annotateRight = Annotater(OMinusESqOverE, ax, plt.gcf())
cidRight = plt.gcf().canvas.mpl_connect('motion_notify_event', annotateRight.onMotion)

plt.figtext(0.5, 0.01, "WARNING: From Scipy Docs: An often quoted guideline for the validity of this calculation is that the test should be used only if the observed and expected frequencies in each cell are at least 5.", wrap=True, horizontalalignment='center', fontsize=12)

plt.show()

plt.gcf().canvas.mpl_disconnect(cidLeft)
plt.gcf().canvas.mpl_disconnect(cidRight)
