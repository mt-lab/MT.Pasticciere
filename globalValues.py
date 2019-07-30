import cv2
import numpy as np
from configValues import cameraAngle, cameraHeight

heightMap = None
distanceToLaser = cameraHeight/np.cos(cameraAngle)
