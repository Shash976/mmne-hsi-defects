#!/usr/bin/python2

import sys
from syntax.vtk_write_legacy import *
from tamu.intensity_stack import *
import numpy as np

# x- and y-direction
fileList = [ "Cluster2_Map.txt", "Cluster3_Map.txt", "Cluster4_Map.txt", "Cluster5_Map.txt"]
# each intensity map is associated with a discrete composition value
factors = [ 0.3, 0.45, 0.1, 0.6 ]

# lower and upper bounds for the composition; depends on the discrete composition
# values associated with the input files as given above
x0 = 0.0
x1 = 0.1

# the physical concentration values corresponding to the composition bounds
# given above; in [mol/m^3]
c0 = 0.0
c1 = 1.85e3 

# parse intensity maps and set up a 2D point grid connected by planar 4-node
# elements for bilinear interpolation
bgMesh,xCrs,yCrs,intensities = readIntensityStack(fileList)
numX = len(xCrs)
numY = len(yCrs)
# apply a Gauss filter to the intensity maps and normalize the result as
# described in the documentation
filtered,summed = filterIntensities(intensities,sig=1,normalize=True)

# as an additional layer of information, parse the thickness map provided by the
# Banerjee group

#f = open("051_thickness.txt","r")
#keyword = getDescriptorString(f)
#thickness = readIntensities(f,keyword).reshape((numY,numX))
#f.close()
# remove noise by application of a Gauss filter
#fThick = ndimage.gaussian_filter(thickness,sigma=1,mode='reflect')

# weighted sum of the intensity maps at each lattice point
xFrac = np.zeros(summed.shape,float)
for i in range(0,filtered.shape[2]):
  xFrac[:,:] += factors[i] * filtered[:,:,i]

# linear interpolation of physical concentration values from computed composition
s = (xFrac - x0) / (x1-x0)
conc = c0 + ((c1-c0)/x1-x0)*s

# associate nodes with raw input data
bgMesh.addNodalDataField("raw_intensity_1",intensities[:,:,0].flatten(),"double")
bgMesh.addNodalDataField("raw_intensity_2",intensities[:,:,1].flatten(),"double")
bgMesh.addNodalDataField("raw_intensity_3",intensities[:,:,2].flatten(),"double")
#bgMesh.addNodalDataField("raw_thickness", thickness.flatten(),"double")
# associate filtered intensity maps
bgMesh.addNodalDataField("filtered_intensity_1",filtered[:,:,0].flatten(),"double")
bgMesh.addNodalDataField("filtered_intensity_2",filtered[:,:,1].flatten(),"double")
bgMesh.addNodalDataField("filtered_intensity_3",filtered[:,:,2].flatten(),"double")
#bgMesh.addNodalDataField("filtered_thickness",fThick.flatten(),"double")
# associate derived map data
bgMesh.addNodalDataField("composition",xFrac.flatten(),"double")
bgMesh.addNodalDataField("concentration",conc.flatten(),"double")
bgMesh.addNodalDataField("order",summed.flatten(),"double")

writeToVTKFileByName("bgGeometry.vtk",bgMesh)
