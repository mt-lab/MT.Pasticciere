import ezdxf as ez
import numpy as np
from utilities import *
from global_variables import *
from math import sqrt, cos, sin, pi


class Element:
    def __init__(self, entity, first=(0, 0), last=(0, 0)):
        self.entity = entity
        self.points = []
        self.first = first
        self.last = last
        self.sliced = []
        self.backwards = False
        self.offset = (0, 0)
        self.length = 0

    def set_offset(self, offset=(0, 0)):
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

    def get_length(self):
        pass

    def slice(self, step=1):
        for start, end in pairwise(self.points):
            for p in diap(start, end, step):
                p = list(p)
                if len(p) != 3:
                    p.append(0)
                elif len(p) > 3:
                    p = p[:3]
                self.sliced.append(p)

    def add_z(self, pcd_xy, pcd_z):
        if len(self.sliced) != 0:
            for p in self.sliced:
                p[Z] = find_point_in_Cloud(p, pcd_xy, pcd_z, self.offset)
        elif len(self.points) != 0:
            for p in self.points:
                p.append(find_point_in_Cloud(p, pcd_xy, pcd_z, self.offset))


class Polyline(Element):
    def __init__(self, polyline):
        super().__init__(polyline)
        self.points = [point for point in polyline.points()]
        self.first = self.points[0]
        self.last = self.points[-1]
        self.sliced = []
        self.length = self.get_length()

    def get_length(self):
        length = 0
        for p1, p2 in pairwise(self.points):
            length += distance(p1, p2)
        return length


class Spline(Element):
    def __init__(self, spline):
        super().__init__(spline)
        self.points = [point for point in spline.control_points]
        self.first = spline.control_points[0]
        self.last = spline.control_points[-1]
        self.sliced = []


class Line(Element):
    def __init__(self, line):
        super().__init__(line)
        self.points = [line.dxf.start, line.dxf.end]
        self.first = self.points[0]
        self.last = self.points[-1]
        self.sliced = []
        self.length = self.get_length()

    def get_length(self):
        return distance(self.first, self.last)


class Circle(Element):
    def __init__(self, circle):
        super().__init__(circle)
        self.center = circle.dxf.center
        self.radius = circle.dxf.radius
        self.start_angle = 0
        self.end_angle = 2 * pi
        self.first = (
            self.center[X] + self.radius * cos(self.start_angle), self.center[Y] + self.radius * sin(self.start_angle))
        self.last = (
            self.center[X] + self.radius * cos(self.end_angle), self.center[Y] + self.radius * sin(self.end_angle))
        self.points = [self.first, self.last]
        self.sliced = []
        self.length = self.get_length()

    def get_length(self):
        return (self.end_angle - self.start_angle) * self.radius

    def slice(self, step=1):
        angle_step = step / self.radius * (self.end_angle - self.start_angle) / abs(
            self.end_angle - self.start_angle)  # в радианах с учетом знака
        for angle in np.arange(self.start_angle, self.end_angle, angle_step):
            p = [self.radius * cos(angle) + self.center[X], self.radius * sin(angle) + self.center[Y], 0]
            self.sliced.append(p)
        last = list(self.last)
        if len(last) < 3:
            last.append(0)
        else:
            last = last[:2]
        self.sliced.append(last)  # дбавление конечной точки в массив нарезанных точек


class Arc(Circle):
    def __init__(self, arc):
        super().__init__(arc)
        self.start_angle = arc.dxf.start_angle * pi / 180  # в радианах
        self.end_angle = arc.dxf.end_angle * pi / 180  # в радианах
        if self.start_angle > self.end_angle:
            self.end_angle += 2 * pi
        self.first = (
            self.center[X] + self.radius * cos(self.start_angle), self.center[Y] + self.radius * sin(self.start_angle))
        self.last = (
            self.center[X] + self.radius * cos(self.end_angle), self.center[Y] + self.radius * sin(self.end_angle))
        self.points = [self.first, self.last]
        self.sliced = []
        self.length = self.get_length()
