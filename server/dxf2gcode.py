import ezdxf as ez
from global_variables import *
from elements import *
from utilities import *
from gcode_gen import *


def dxf_reader(dxf, modelspace, elements_heap=[]):  # at first input is modelspace
    for element in modelspace:
        if element.dxftype() == 'INSERT':
            block = dxf.blocks[element.dxf.name]
            dxf_reader(dxf, block, elements_heap)
        else:
            elements_heap.append(element)
    return elements_heap


def dxf2gcode(pathToDxf, pathToPLY, offset=(0, 0)):
    PATH_TO_DXF = pathToDxf
    cloud = pathToPLY
    dxf = ez.readfile(PATH_TO_DXF)
    msp = dxf.modelspace()
    # get all entities from dxf
    entities = dxf_reader(dxf, msp)
    # read pcd
    pcd, pcd_xy, pcd_z = read_pcd(cloud)

    # for easier work with elements define them with elements class
    unproc = []
    for e in entities:
        if e.dxftype() == 'POLYLINE':
            unproc.append(Polyline(e))
        elif e.dxftype() == 'SPLINE':
            unproc.append(Spline(e))
        elif e.dxftype() == 'LINE':
            unproc.append(Line(e))
        elif e.dxftype() == 'CIRCLE':
            unproc.append(Circle(e))
        elif e.dxftype() == 'ARC':
            unproc.append(Arc(e))

    # organize path
    path = []
    unproc.sort(key=lambda x: x.best_distance((0, 0)))
    while len(unproc) != 0:
        cur = unproc[0]
        path.append(cur)
        unproc.pop(0)
        unproc.sort(key=lambda x: x.best_distance(cur.get_points()[-1]))

    # slice dxf and add volume to it, also add offset
    for element in path:
        element.slice(step)
        element.set_offset(offset)
        element.add_z(pcd_xy, pcd_z)

    # generate gcode
    gcode = gcode_generator(path)

    # write gcode
    with open('cookie.gcode', 'w+') as gfile:
        for line in gcode:
            gfile.write(line + '\n')
