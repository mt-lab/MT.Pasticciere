import ezdxf as ez
from utilities import *
from math import sqrt

X, Y, Z = 0, 1, 2


class Element:
    def __init__(self, entity, first=(), last=()):
        self.entity = entity
        self.points = []
        self.first = first
        self.last = last
        self.sliced = []
        self.backwards = False
        self.offset = (0, 0)

    def set_offset(self, offset=(0,0)):
        if len(self.sliced) != 0:
            for point in self.sliced:
                point[X] -= self.offset[X]
                point[X] += offset[X]
                point[Y] -= self.offset[Y]
                point[Y] += offset[Y]
            self.offset = offset
        else:
            print('nothing to offset')


    def best_distance(self, point):
        dist_to_first = sqrt(abs(self.first[X] - point[X]) ** 2 + abs(self.first[Y] - point[Y]) ** 2)
        dist_to_last = sqrt(abs(self.last[X] - point[X]) ** 2 + abs(self.last[Y] - point[Y]) ** 2)
        self.backwards = dist_to_last < dist_to_first
        return min(dist_to_first, dist_to_last)

    def get_points(self):
        return self.points if not self.backwards else self.points[::-1]

    def get_sliced_points(self):
        return self.sliced if not self.backwards else self.sliced[::-1]

    def slice(self, step=1):
        for start, end in pairwise(self.points):
            for p in diap(start, end, step):
                p = list(p)
                if len(p) !=3:
                    p.append(0)
                self.sliced.append(p)

    def add_z(self, pcd_xy, pcd_z):
        if len(self.sliced) != 0:
            for p in self.sliced:
                p[Z] = find_point_in_Cloud(p, pcd_xy, pcd_z, self.offset)
        elif len(self.points) != 0:
            for p in self.points:
                p.append(find_point_in_Cloud(p, pcd_xy, pcd_z, self.offset))


class Polyine(Element):
    def __init__(self, polyline):
        super().__init__(polyline)
        self.points = [point for point in polyline.points()]
        self.first = self.points[0]
        self.last = self.points[-1]
        self.sliced = []


class Spline(Element):
    def __init__(self, spline):
        super().__init__(spline)
        self.points = [point for point in spline.control_points]
        self.first = spline.control_points[0]
        self.last = spline.control_points[-1]
        self.sliced = []


class Line(Element):
    pass


class Arc(Element):
    pass


class Circle(Element):
    pass
