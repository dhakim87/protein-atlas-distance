import csv;
import sys;
import math;
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.odr
from scipy.stats import pearsonr

COMBINED_DIST_FILE = "u2os_similarity_combined.csv"

#Linear Fit, B[0] is slope, B[1] is y intercept
def fitFunction(p, x):
   m, c = p
   return m*x + c

def runTotalLeastSquaresRegression(x,y):
    # Create a model for fitting.
    linear_model = scipy.odr.Model(fitFunction)

    # Create a RealData object using our initiated data from above.
    data = scipy.odr.RealData(x, y)

    # Set up ODR with the model and data.
    odr = scipy.odr.ODR(data, linear_model, beta0=[0., 1.])

    # Run the regression.
    out = odr.run()

    # Use the in-built pprint method to give us results.
    out.pprint()

    return out.beta

parseHeader = True
numRowsRead = 0
with open(COMBINED_DIST_FILE) as csvFile:
    xArr = []
    yArr = []
    reader = csv.reader(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        numRowsRead += 1
        if numRowsRead % 100000 == 0:
            print numRowsRead
        if numRowsRead == 100000000:
            print "BAIL!"
            break

        if parseHeader:
            parseHeader = False
            header = row
            continue

        xArr.append(float(row[2]))
        yArr.append(float(row[3]))

print "Creating Numpy Arrays"
# Create data
x = np.array(xArr)
y = np.array(yArr)

print "Running Linear Regressions..."
#Total least squares (distance computed perpendicular to line) - This takes forever!
#tlsSlopeAndIntercept = runTotalLeastSquaresRegression(xArr, yArr)
#Ordinary Linear Regression (OLS, Ordinary Least Squares, distance computed vertical from line)
polyCoeffs = np.polyfit(x,y,1)

print "Histogramming"
xIn = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2,1.3,1.4,1.5]
yIn = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]
(H, xedges, yedges) = np.histogram2d(x,y,bins=(xIn, yIn))
H = H.T
xLabels = []
yLabels = []
prev = xedges[0]
for nextX in xedges[1:]:
    xLabels.append("%.2f" % ((prev + nextX)/2))
    prev = nextX
prev = yedges[0]
for nextY in yedges[1:]:
    yLabels.append("%.2f" % ((prev + nextY)/2))
    prev = nextY
ax = sns.heatmap(H, linewidth=0.5, xticklabels=xLabels, yticklabels=yLabels, square=True)
ax.set_xlabel("Dan Dist")
ax.set_ylabel("Yue Similarity")


#Plot linear regression on top of heatmap

#realSpacePoints
P1 = (xedges[0], xedges[0] * polyCoeffs[0] + polyCoeffs[1])
P2 = (xedges[-1], xedges[-1] * polyCoeffs[0] + polyCoeffs[1])

#P3 = (xedges[0], xedges[0] * tlsSlopeAndIntercept[0] + tlsSlopeAndIntercept[1])
#P4 = (xedges[-1], xedges[-1] * tlsSlopeAndIntercept[0] + tlsSlopeAndIntercept[1])

#heatmapSpacePoints
p1x = (P1[0] - xedges[0]) / (xedges[-1] - xedges[0]) * (len(xedges)-1)
p1y = (P1[1] - yedges[0]) / (yedges[-1] - yedges[0]) * (len(yedges)-1)
p2x = (P2[0] - xedges[0]) / (xedges[-1] - xedges[0]) * (len(xedges)-1)
p2y = (P2[1] - yedges[0]) / (yedges[-1] - yedges[0]) * (len(yedges)-1)
#p3x = (P3[0] - xedges[0]) / (xedges[-1] - xedges[0]) * (len(xedges)-1)
#p3y = (P3[1] - yedges[0]) / (yedges[-1] - yedges[0]) * (len(yedges)-1)
#p4x = (P4[0] - xedges[0]) / (xedges[-1] - xedges[0]) * (len(xedges)-1)
#p4y = (P4[1] - yedges[0]) / (yedges[-1] - yedges[0]) * (len(yedges)-1)



print "Poly Coeffs: ", polyCoeffs
pearsonCoeff, pearsonPVal = pearsonr(x, y)
print "Pearson Coeff: ", pearsonCoeff
print "Pearson p-value: ", pearsonPVal


ax.plot([p1x, p2x],[p1y, p2y])
#ax.plot([p3x, p4x],[p3y, p4y])
#Debug, draw line across the heatmap in heatmap space.
#ax.plot([0,len(xedges)-1], [0,len(yedges)-1])

plt.show()


#print "Scatterplotting"
## Plot
#plt.scatter(x, y)
#plt.xlabel('Dan Dist')
#plt.ylabel('Yue Dist')
#print "Showing Plot"
plt.show()
