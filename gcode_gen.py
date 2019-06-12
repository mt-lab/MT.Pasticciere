import math as m
import ezdxf as ez
from pygcode import *
from global_variables import *
from utilities import *
import random
import numpy as np
from elements import *

extr_coef = 0.41
Z_up = Z_max + 3  # later should be cloud Z max + few mm


# сейчас это глобальный максимум печати принтера по Z

# when path is a set of LINES
# DO NOT USE! TO BE DELETED
def generate_gcode(path):
    gcode = []
    curr = [0, 0, 0]
    for fig in path:
        start = fig[0]  # list(fig.dxf.start)
        end = fig[1]  # list(fig.dxf.end)
        if len(start) == 2:
            start.append(0)
            end.append(0)
        distance = m.sqrt((start[X] - curr[X]) ** 2 + (start[Y] - curr[Y]) ** 2 + (start[Z] - curr[Z]) ** 2)
        print(distance)
        if distance < error:
            gcode.append(str(GCodeLinearMove(X=end[X], Y=end[Y], Z=end[Z])))
        else:
            gcode.append(str(GCodeRapidMove(Z=Z_up)))
            gcode.append(str(GCodeRapidMove(X=start[X], Y=start[Y], Z=start[Z])))
            gcode.append(str(GCodeLinearMove(X=end[X], Y=end[Y], Z=end[Z])))
        curr = end
    return gcode


# when path is a set of elements
def gcode_generator(path):
    gcode = []
    last_point = (0, 0, 0)
    E = 0
    for count, element in enumerate(path):
        way = element.get_sliced_points()
        gcode.append('; %03d element' % (count + 1))
        if distance(last_point, way[0]) > error:
            gcode.append(str(GCodeRapidMove(Z=Z_up)))
            gcode.append(str(GCodeRapidMove(X=way[0][X], Y=way[0][Y])))
            gcode.append(str(GCodeRapidMove(Z=way[0][Z])))
            last_point = way[0]
        for point in way[1:]:
            E += round(extr_coef*distance(last_point, point),3)
            gcode.append(str(GCodeLinearMove(X=point[X], Y=point[Y], Z=point[Z])) + ' E%03f' % E)
            last_point = point
        last_point = way[-1]
    return gcode
