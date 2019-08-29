import numpy as np
from configValues import cameraAngle, cameraHeight
from utilities import readHeightMap

heightMap = readHeightMap()
cookies = None
distanceToLaser = cameraHeight/np.cos(cameraAngle)
