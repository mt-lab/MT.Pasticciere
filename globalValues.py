import cv2
import numpy as np
from configValues import cameraAngle, cameraHeight

heightMap = np.empty((480,640), dtype=np.float64)
distanceToLaser = cameraHeight/np.cos(cameraAngle)
