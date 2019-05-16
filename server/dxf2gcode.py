import ezdxf as ez
from elements import *
from utilities import *
from gcode_gen import *


class Unwrap_dxf:
    """ Class to get elements from dxf """

    def __init__(self, dxf):
        self.dxf = dxf
        self.msp = dxf.modelspace()
        self.entities = []

    def unwrap(self):
        """ Returns set of all elements in dxf """
        for e in self.msp:
            self.block_unwrap(e)
        return self.entities

    def block_unwrap(self, entity):
        """ Process nested blocks """
        if entity.dxftype() == 'INSERT':
            block = self.dxf.blocks[entity.dxf.name]
            for e in block:
                self.block_unwrap(e)
        else:
            self.entities.append(entity)


def dxf2gcode(pathToDxf):
    PATH_TO_DXF = pathToDxf
    cloud = '../scanner/cloud.ply'
    dxf = ez.readfile(PATH_TO_DXF)
    # get all entities from dxf
    entities = Unwrap_dxf(dxf).unwrap()
    # read pcd
    pcd, pcd_xy, pcd_z = read_pcd(cloud)

    # for easier work with elements define them with elements class
    unproc = []
    for e in entities:
        if e.dxftype() == 'POLYLINE':
            unproc.append(Polyine(e))
        elif e.dxftype() == 'SPLINE':
            unproc.append(Spline(e))

    # organize path
    path = []
    unproc.sort(key=lambda x: x.best_distance((0, 0)))
    while len(unproc) != 0:
        cur = unproc[0]
        path.append(cur)
        unproc.pop(0)
        unproc.sort(key=lambda x: x.best_distance(cur.get_points()[-1]))

    # slice dxf and add volume to it
    step = 1
    for element in path:
        element.slice(step)
        element.add_z(pcd_xy, pcd_z)

    # generate gcode
    gcode = gcode_generator(path)

    # write gcode
    with open('coockie.gcode', 'w+') as gfile:
        for line in gcode:
            gfile.write(line + '\n')
