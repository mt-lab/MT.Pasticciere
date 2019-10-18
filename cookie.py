from math import pi
from utilities import X, Y, Z
import numpy as np
import cv2


class Cookie:
    def __init__(self, height_map, contour, bounding_box=None):
        self._contour = contour  # контур из целой карты высот
        self._contour_mm = None
        self._height_map = height_map  # область печеньки на карте высот
        self._center = None  # центр печенья в мм (X, Y, Z)
        self._pixel_center = None  # центр печенья в пикселях на целой карте высот
        self._bounding_box = bounding_box  # центр, высота и ширина области печеньки на карте высот
        self._bounding_box_mm = None
        self._min_bounding_box = None
        self._min_bounding_box_mm = None
        self._width = None  # размер печеньки вдоль Y оси в мм
        self._length = None  # размер печеньки вдоль X оси в мм
        self._max_height = None  # максимальная высота в контуре ограничивающем печенье в мм
        self._rotation = None  # ориентация печеньки в пространстве в радианах

    @property
    def height_map(self):
        return self._height_map

    @property
    def contour(self):
        return self._contour

    @property
    def contour_mm(self):
        if self._contour_mm is None:
            col, row, _, _ = self.bounding_box
            self._contour_mm = np.asarray([(self.height_map[point[0][1] - row, point[0][0] - col, Y],
                                            self.height_map[point[0][1] - row, point[0][0] - col, X]) for point in
                                           self.contour],
                                          dtype=np.float32)
        return self._contour_mm

    @property
    def center(self):
        if self._center is None:
            self.find_center_and_rotation()
        return self._center

    @property
    def rotation(self):
        if self._rotation is None:
            self.find_center_and_rotation()
        return self._rotation

    @property
    def width(self):
        if self._width is None:
            self._width = self.min_bounding_box_mm[1][1]
        return self._width

    @property
    def length(self):
        if self._length is None:
            self._length = self.min_bounding_box_mm[1][0]
        return self._length

    @property
    def pixel_center(self):
        if self._pixel_center is None:
            moments = cv2.moments(self.contour)
            col = int(round(moments['m10'] / moments['m00']))
            row = int(round(moments['m01'] / moments['m00']))
            self._pixel_center = (row, col)
        return self._pixel_center

    @property
    def bounding_box(self):
        if self._bounding_box is None:
            self._bounding_box = cv2.boundingRect(self.contour)
        return self._bounding_box

    @property
    def bounding_box_mm(self):
        if self._bounding_box_mm is None:
            self._bounding_box_mm = cv2.boundingRect(self.contour_mm)
        return self._bounding_box_mm

    @property
    def min_bounding_box(self):
        if self._min_bounding_box is None:
            self._min_bounding_box = cv2.minAreaRect(self.contour)
        return self._min_bounding_box

    @property
    def min_bounding_box_mm(self):
        if self._min_bounding_box_mm is None:
            self._min_bounding_box_mm = cv2.minAreaRect(self.contour_mm)
        return self._min_bounding_box_mm

    @property
    def max_height(self):
        if self._max_height is None:
            self._max_height = self.height_map[:, :, Z].max()
        return self._max_height

    def find_center_and_rotation(self):
        # TODO: допилить под нахождение true центра
        center_row, center_col = self.pixel_center
        col, row, w, h = self.bounding_box  # кординаты описывающего прямоугольника
        center_z = self.height_map[center_row - row, center_col - col, Z]
        # Найти центр и поворот контура по точкам в мм
        moments = cv2.moments(self.contour_mm)
        center_x = moments['m10'] / moments['m00']
        center_y = moments['m01'] / moments['m00']
        a = moments['m20'] / moments['m00'] - center_x ** 2
        b = 2 * (moments['m11'] / moments['m00'] - center_x * center_y)
        c = moments['m02'] / moments['m00'] - center_y ** 2
        theta = 1 / 2 * np.arctan(b / (a - c)) + (a < c) * pi / 2
        center = (center_y, center_x, center_z)  # координаты свапнуты из-за opencv
        rotation = pi / 2 - theta  # перевод в СК принтера
        self._center = center
        self._rotation = rotation

    def __str__(self):
        return 'Центр:\n' + \
               f'    X: {self.center[X]: 4.2f} мм\n' + \
               f'    Y: {self.center[Y]: 4.2f} мм\n' + \
               f'    Z: {self.center[Z]: 4.2f} мм\n' + \
               f'Поворот: {self.rotation * 180 / pi:4.2f} градусов\n' + \
               f'Максимальная высота: {self.max_height:4.2f} мм'
