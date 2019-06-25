import math as m
import ezdxf as ez
from pygcode import *
from global_variables import *
from utilities import *
import random
import numpy as np
from elements import *

Z_up = Z_max + 3  # later should be cloud Z max + few mm сейчас это глобальный максимум печати принтера по Z
extrusionCoefficient = 0.41


# when path is a set of elements
def gcode_generator(path, preGcode=[], postGcode=[]):
    # TODO: префиксный и постфиксный Gcode
    # TODO: генерация кода по печенькам
    # TODO: перенести в файл dxf2gcode.py
    gcode = []
    last_point = (0, 0, 0)
    E = 0
    gcode.append('G28')
    for count, element in enumerate(path, 1):
        way = element.getSlicedPoints()
        gcode.append(f'; {count:3d} element')
        if distance(last_point, way[0]) > accuracy:
            gcode.append(str(GCodeRapidMove(Z=Z_up)))
            gcode.append(str(GCodeRapidMove(X=way[0][X], Y=way[0][Y])))
            gcode.append(str(GCodeRapidMove(Z=way[0][Z])))
            last_point = way[0]
        for point in way[1:]:
            E += round(extrusionCoefficient * distance(last_point, point), 3)
            gcode.append(str(GCodeLinearMove(X=point[X], Y=point[Y], Z=point[Z])) + f' E{E:3.3f}')
            last_point = point
        last_point = way[-1]
    return gcode
