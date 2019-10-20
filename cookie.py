from math import pi
from typing import Union
from utilities import X, Y, Z, find_center_and_rotation, find_contours
import numpy as np
import cv2


class Cookie:
    def __init__(self, height_map, contour, bounding_box=None):
        self._contour = contour  # контур из целой карты высот (col, row)
        self._contour_local = None  # контур в локальной карте высот (col, row)
        self._contour_mm = None  # контур в мм в глобальных координатах (Y,X)
        self._height_map = height_map  # область печеньки на карте высот
        self._center = None  # центр контура печенья в мм (X, Y, Z)
        self._center_true = None  # центр печенья с учетом искажения сканирования (X, Y, Z)
        self._center_pixel = None  # центр печенья в пикселях на целой карте высот (col, row)
        self._center_local = None  # центр печенья в пикселях на локальной карте высот (col, row)
        self._rotation = None  # ориентация печеньки в пространстве в радианах
        self._rotation_true = None
        self._bounding_box = bounding_box  # центр, высота и ширина области печеньки на карте высот (col, row, w, h)
        self._bounding_box_mm = None  # (Y, X, w, h) в мм
        self._min_bounding_box = None  # ((col, row), (w, h), angle_deg) пиксели
        self._min_bounding_box_mm = None  # ((Y, X), (w, h), angle_deg) в мм
        self._width = None  # shorthand for min_bounding_box_mm[1,0] в мм
        self._length = None  # shorthand for min_bounding_box_mm[1,1] в мм
        self._max_height = None  # максимальная высота в контуре ограничивающем печенье в мм

    @property
    def height_map(self):
        return self._height_map

    @property
    def contour(self):
        return self._contour

    @property
    def contour_local(self):
        if self._contour_local is None:
            col, row, *_ = self.bounding_box
            self._contour_local = self.contour - np.array([col, row])
        return self._contour_local

    @property
    def contour_mm(self):
        if self._contour_mm is None:
            self._contour_mm = np.asarray([[(self.height_map[point[0, 1], point[0, 0], Y],
                                             self.height_map[point[0, 1], point[0, 0], X])]
                                           for point in self.contour_local], dtype=np.float32)
        return self._contour_mm

    @property
    def center(self):
        if self._center is None:
            self.find_center_and_rotation()
        return self._center

    @property
    def center_pixel(self):
        if self._center_pixel is None:
            self._center_pixel = find_center_and_rotation(self.contour, False)
            self._center_pixel = tuple(int(round(c)) for c in self._center_pixel)
        return self._center_pixel

    @property
    def center_local(self):
        if self._center_local is None:
            self._center_local = (self.center_pixel[0] - self.bounding_box[0],
                                  self.center_pixel[1] - self.bounding_box[1])
        return self._center_local

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

    @property
    def contour_idx(self) -> tuple:
        return tuple(np.hsplit(np.fliplr(self.contour_local.reshape(self.contour_local.shape[0], 2)), 2))

    @property
    def contour_coords(self) -> np.ndarray:
        return self.height_map[self.contour_idx]

    @property
    def contour_mean(self, axis=None):
        return self.contour_coords.mean(axis)

    @property
    def contour_std(self, axis=None):
        return self.contour_coords.std(axis)

    def find_center_and_rotation(self):
        # TODO: допилить под нахождение true центра
        center_col, center_row = self.center_local
        center_z = self.height_map[center_row, center_col, Z]
        # Найти центр и поворот контура по точкам в мм
        center, theta = find_center_and_rotation(self.contour_mm)
        center = (*center[::-1], center_z)  # координаты свапнуты из-за opencv
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


def find_cookies(img: Union[np.ndarray, str], height_map: 'np.ndarray'):
    """
    Функция нахождения расположения и габаритов объектов на столе из полученной карты высот
    :param Union[np.ndarray, str] img: карта высот
    :param np.ndarray height_map: облако точек соответствующее карте высот
    :return cookies, result: параметры печенек, картинка с визуализацией, параметры боксов
            ограничивающих печеньки, контура границ печенья
    """

    contours, result = find_contours(img)

    cookies = []
    for contour in contours:
        mask = np.zeros(height_map.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        col, row, w, h = cv2.boundingRect(contour)
        height_map_masked = height_map.copy()
        height_map_masked[..., Z][mask == 0] = 0
        height_map_fragment = height_map_masked[row:row + h, col:col + w]
        cookie = Cookie(height_map=height_map_fragment, contour=contour, bounding_box=(col, row, w, h))
        cookies.append(cookie)
        cv2.circle(result, cookie.center_pixel, 3, (0, 255, 0), -1)

    print('Положения печений найдены.')
    return cookies, result
