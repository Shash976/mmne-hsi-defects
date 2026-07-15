#!/usr/bin/python2


import sys


from gmsh.gmsh_model import *
from syntax.vtk_write_legacy import *
from syntax.gmsh_write_geo import *
from syntax.gmsh_read_msh import parseMshFile
from tamu.intensity_stack import *
from scipy import misc,ndimage
from skimage import measure
import numpy as np



x0 = 0.0
x1 = 0.1

c0 = 0.0
c1 = 1.85e3

overlayMeshSize = 0.25

def writeOutlineGeoFile(order,numX,numY,threshold):
  contours = measure.find_contours(order.reshape((numY,numX)),threshold)
  numOverlayPoints = 0
  for item in contours:
    numOverlayPoints += len(item)
  outCoors = np.zeros((numOverlayPoints,2),float)
  index = 0
  for item in contours:
    outCoors[index:index+len(item),:] = item[:,:]
    index += len(item)

  outlineIncidence = np.zeros((numOverlayPoints,2),int)
  index = 0
  last = numOverlayPoints
  for i in range(0,numOverlayPoints):
    outlineIncidence[index,0] = last
    outlineIncidence[index,1] = last - 1
    last -= 1
    index += 1
  outlineIncidence[-1,1] = numOverlayPoints

  model = GmshModel()
  outFile = "overlay_threshold_%1.2f.geo"%threshold
  for i in range(0,numOverlayPoints):
    model.addPoint_Coordinates(outCoors[i,1],outCoors[i,0],0.0,size=overlayMeshSize)
  outlineLineLoop = np.arange(index,index+numOverlayPoints+1)
  lines = []
  for i in range(0,numOverlayPoints):
    lines.append(model.addLine(outlineIncidence[i,0],outlineIncidence[i,1]))
  ll = model.addLineLoop(lines)
  model.addPlaneSurface(ll)
  writeGeoFile(outFile,model)



fileList = [ "PC2_text_file.txt", "PC3_text_file.txt", "PC4_text_file.txt" ]
factors = [ 0.07, 0.0, 0.1 ]

bgMesh,xCrs,yCrs,intensities = readIntensityStack(fileList)
numX = len(xCrs)
numY = len(yCrs)
filtered,summed = filterIntensities(intensities,sig=1)

f = open("051_thickness.txt","r")
keyword = getDescriptorString(f)
thickness = readIntensities(f,keyword).reshape((numY,numX))
f.close()
fThick = ndimage.gaussian_filter(thickness,sigma=1,mode='reflect')

xFrac = np.zeros(summed.shape,float)
for i in range(0,filtered.shape[2]):
  xFrac[:,:] += factors[i] * filtered[:,:,i]

s = (xFrac - x0) / (x1-x0)
conc = c0 + ((c1-c0)/x1-x0)*s

# extract nanowire outline for given threshold
##############################################
writeOutlineGeoFile(summed,numX,numY,threshold=0.08)

