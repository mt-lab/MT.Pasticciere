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
        elif element_redef(element):
            elements_heap.append(element_redef(element))
        else:
            print('empty element')
    return elements_heap


def element_redef(element):
    if element.dxftype() == 'POLYLINE':
        return Polyline(element)
    elif element.dxftype() == 'SPLINE':
        return Spline(element)
    elif element.dxftype() == 'LINE':
        return Line(element)
    elif element.dxftype() == 'CIRCLE':
        return Circle(element)
    elif element.dxftype() == 'ARC':
        return Arc(element)
    elif element.dxftype() == 'ELLIPSE':
        pass
    elif element.dxftype() == 'LWPOLYLINE':
        pass
    elif element.dxftype() == 'POINT':
        pass
    # else:
    #     return Element(element)


def organize_path(elements, start_point=(0, 0)):
    path = []
    elements.sort(key=lambda x: x.best_distance(start_point))
    while len(elements) != 0:
        cur = elements[0]
        path.append(cur)
        elements.pop(0)
        elements.sort(key=lambda x: x.best_distance(cur.get_points()[-1]))
    return path


def process_path(path, offset=(0, 0), pathToPLY=PCD_PATH):
    # read pcd
    pcd, pcd_xy, pcd_z = read_pcd(pathToPLY)
    # slice dxf and add volume to it, also add offset
    for element in path:
        element.slice(step)
        element.set_offset(offset)
        element.add_z(pcd_xy, pcd_z)


def write_gcode(gcode_instructions, filename='cookie.gcode'):
    with open(filename, 'w+') as gcode:
        for line in gcode_instructions:
            gcode.write(line + '\n')


def dxf2gcode(pathToDxf=DXF_PATH, pathToPLY=PCD_PATH, offset=(0, 0)):
    dxf = ez.readfile(pathToDxf)
    msp = dxf.modelspace()
    # get all entities from dxf
    elements_heap = dxf_reader(dxf, msp)

    # organize path
    path = organize_path(elements_heap)

    # slice dxf and add volume to it, also add offset
    process_path(path, offset, pathToPLY)

    # generate gcode instructions as array
    gcode_instructions = gcode_generator(path)

    # write gcode to text file
    write_gcode(gcode_instructions)
