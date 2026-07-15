#!/usr/bin/python2

from syntax.vtk_write_legacy import *
from syntax.gmsh_read_msh import parseMshFile
from syntax.feap_write_mesh import *
from tamu.intensity_stack import *
import numpy as np

# the mesh file for whose nodes we want to determine the nodal concentrations
meshFile = "bgGeometry.vtk"

x0 = 0.1
x1 = 0.4

# the physical concentration values corresponding to the composition bounds
# given above; in [mol/m^3]
c0 = 1.85E+03
c1 = 7.26E+03

# determine the coordinate range i such that entries_i <= value <= entries_{i+1}
def findSpan(entries,value):
  index = 0
  if(abs(entries[-1]-value) <= 1e-7):
    return (len(entries) - 2)
  while(entries[index] <= value):
    index += 1
  return (index - 1)

def getShapeFunctions(shp,sg):
  shp.fill(0.0)
  oneMinusR = 1.0 - sg[0]
  oneMinusS = 1.0 - sg[1]
  onePlusR = 1.0 + sg[0]
  onePlusS = 1.0 + sg[1]
  shp[0] = 0.25 * oneMinusR * oneMinusS
  shp[1] = 0.25 * onePlusR * oneMinusS
  shp[2] = 0.25 * onePlusR * onePlusS
  shp[3] = 0.25 * oneMinusR * onePlusS

#fileList = [ "PC2_text_file.txt", "PC3_text_file.txt", "PC4_text_file.txt" ]
#factors = [ 0.07, 0.0, 0.1 ]

fileList = [ "TXT_alpha_01_corrected.txt","TXT_epsilon_04_corrected.txt" ]
factors = [ 0.1, 0.4 ]

bgMesh,xCrs,yCrs,intensities = readIntensityStack(fileList)
numX = len(xCrs)
numY = len(yCrs)
filtered,summed = filterIntensities(intensities,sig=1)

xFrac = np.zeros(summed.shape,float)
for i in range(0,filtered.shape[2]):
  xFrac[:,:] += factors[i] * filtered[:,:,i]

s = (xFrac - x0) / (x1-x0)
conc = c0 + ((c1-c0)/x1-x0)*s

# map field values onto FE mesh
###############################

overlayMesh = parseMshFile(meshFile,reportLevel=SILENT)
overlayCoors = overlayMesh.glCoors

# for meshes obtained from the Marching Cubes algorithm on the pixel data, it
# might be necessary to transform the nodal coordinates back to the physical
# coordinates (as the MC algorithm might return pixel coordinates instead of
# physical coordinates). In that case, uncomment the following block
# overlayCoors[:,0] *= (1.0 / (numX-1))
# overlayCoors[:,1] *= (1.0 / (numY-1))
# xMin = xCrs[0]
# xMax = xCrs[-1]
# overlayCoors[:,0] *= (xMax-xMin)
# overlayCoors[:,0] += xMin
# yMin = yCrs[0]
# yMax = yCrs[-1]
# overlayCoors[:,1] *= (yMax-yMin)
# overlayCoors[:,1] += yMin

numOverlayPoints = len(overlayCoors)

bcFlag = np.zeros((numOverlayPoints,3),int)
bcVals = np.zeros((numOverlayPoints,3),float)

bcFlag[:,2] = 1
xMin = np.amin(overlayCoors[:,0])
yMin = np.amin(overlayCoors[:,1])

xMax = np.amax(overlayCoors[:,0])
yMax = np.amax(overlayCoors[:,1])

# if the nodes are situated on the image space boundaries, restrict their
# displacement normal to that boundary - adapt as necessary for the specific
# configuration you're investigating!

for i in range(0,numOverlayPoints):
  if(abs(overlayCoors[i,0]-xMin)<1e-7):
   bcFlag[i,0] = 1
  if(abs(overlayCoors[i,1]-yMin)<1e-7):
    bcFlag[i,1] = 1
  if(abs(overlayCoors[i,0]-xMax)<1e-7):
    bcFlag[i,0] = 1
  if(abs(overlayCoors[i,1]-yMax)<1e-7):
    bcFlag[i,1] = 1

bgCoors = bgMesh.glCoors
shp = np.zeros(4,float)
sg = np.zeros(2,float)
testPt = np.zeros(2,float)
diff = np.zeros(2,float)
drdx = np.zeros((2,2),float)
dxdr = np.zeros((2,2),float)
xInLixV2O5 = np.zeros(numOverlayPoints,float)
xFrac = xFrac.flatten()
conc = conc.flatten()

# mapping of nodal values onto the "overlay mesh": for each point of that  mesh,
# determine its position w.r.t. 'bgMesh' and compute from this the corresponding
# concentration
for i in range(0,numOverlayPoints):
  pt = overlayCoors[i,0:2]
  xSpan = findSpan(xCrs,pt[0])
  ySpan = findSpan(yCrs,pt[1])
  # determine the parametric coordinates of the nodal point; works like this
  # only because we have a regular, undistorted background mesh
  xSlope = 0.5*(xCrs[xSpan+1] - xCrs[xSpan])
  ySlope = 0.5*(yCrs[ySpan+1] - yCrs[ySpan])
  xAvg = 0.5*(xCrs[xSpan+1] + xCrs[xSpan])
  yAvg = 0.5*(yCrs[ySpan+1] + yCrs[ySpan])
  sg[0] = (pt[0] - xAvg)/xSlope
  sg[1] = (pt[1] - yAvg)/ySlope
  # index of the element we're currently in
  index = xSpan + ySpan*(numX-1)
  cellType = bgMesh.cells[index,0]
  # pointer to the first node index for this element
  ptr = bgMesh.cells[index,1]
  ptsPerCell = nodesPerCell_VTK[cellType]
  # extract the node indices defining this element
  incidence = bgMesh.cellIncidence[ptr:ptr+ptsPerCell]
  # extract the nodal concentration/composition values
  cellConc = conc[incidence]
  compos = xFrac[incidence]
  getShapeFunctions(shp,sg)
  # bilinear interpolation via the shape functions
  for j in range(0,4):
    bcVals[i,2] += cellConc[j] * shp[j]
    xInLixV2O5[i] += compos[j] * shp[j]

# optional output of the overlay mesh w/ mapped data
overlayMesh.addNodalDataField("conc",bcVals[:,2],"double")
overlayMesh.addNodalDataField("comp",xInLixV2O5,"double")
writeToVTKFileByName("nanowire.vtk",overlayMesh)

# output as FEAP mesh
msh = open("mesh.dat","w")
writeNodalCoordinates(msh,overlayMesh.glCoors)
volCells = overlayMesh.extractCellsByType(VTK_TRIANGLE)
msh.write("\nElements")
elmtStartIndex = 1
elmtStartIndex = writeCellGroupToFeap(msh,volCells,elmtStartIndex,1)
msh.close()

# prescription of mapped concentration values as fixed values for the FE
# calculations
bc = open("bc.dat","w")
bc.write("BOUN\n")
for i in range(0,numOverlayPoints):
  bc.write("  %u 0  %u %u %u\n"%(i+1,bcFlag[i,0],bcFlag[i,1],bcFlag[i,2]))
bc.write("\nDISP\n")
for i in range(0,numOverlayPoints):
  bc.write("  %u 0  %f %f %f\n"%(i+1,bcVals[i,0],bcVals[i,1],bcVals[i,2]))
bc.close()
