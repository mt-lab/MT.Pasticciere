import cv2
import numpy as np
from configValues import cameraAngle, cameraHeight
from utilities import readHeightMap

heightMap = readHeightMap()
distanceToLaser = cameraHeight/np.cos(cameraAngle)
