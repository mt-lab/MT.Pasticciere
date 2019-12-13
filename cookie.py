from math import pi
from typing import Union, List
from utilities import X, Y, Z, find_center_and_rotation, find_contours, show_height_map, mls_height_apprx, normalize
import numpy as np
import cv2


class Cookie:
    def __init__(self, height_map, contour_global, bounding_box=None, contour_center=None):
        self._contour_global = contour_global  # контур из целой карты высот (col, row)
        self._contour_local = None  # контур в локальной карте высот (col, row)
        self._contour_center = contour_center
        # TODO: изменить координаты на (X,Y) для избежания путаницы с углом поворота печенья
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
        self._area = None
        self._max_height = None  # максимальная высота в контуре ограничивающем печенье в мм

    @property
    def height_map(self) -> np.ndarray:
        return self._height_map

    @property
    def contour_global(self) -> np.ndarray:
        return self._contour_global

    @property
    def contour_local(self) -> np.ndarray:
        col, row, *_ = self.bounding_box
        return self.contour_global - np.array([col, row])

    @property
    def contour_center(self) -> np.ndarray:
        if self._contour_center is None:
            return self.contour_global
        return self._contour_center

    @property
    def contour_center_local(self) -> np.ndarray:
        col, row, *_ = self.bounding_box
        return self.contour_center - np.array([col, row])

    @contour_center.setter
    def contour_center(self, contour):
        self._contour_center = contour
        self._center = None

    def contour_mm(self, contour='local') -> np.ndarray:
        if contour == 'local':
            contour = self.contour_local
        elif contour == 'center':
            contour = self.contour_center_local
        else:
            raise Exception
        return np.asarray([[(self.height_map[point[0, 1], point[0, 0], Y],
                             self.height_map[point[0, 1], point[0, 0], X])]
                           for point in contour], dtype=np.float32)

    @property
    def center(self) -> tuple:
        if self._center is None:
            self.find_center_and_rotation()
        return self._center

    @property
    def center_pixel(self) -> tuple:
        self._center_pixel = find_center_and_rotation(self.contour_center, False)
        self._center_pixel = tuple(int(round(c)) for c in self._center_pixel)
        return self._center_pixel

    @property
    def center_local(self) -> tuple:
        self._center_local = (self.center_pixel[0] - self.bounding_box[0],
                              self.center_pixel[1] - self.bounding_box[1])
        return self._center_local

    @property
    def rotation(self) -> float:
        if self._rotation is None:
            self.find_center_and_rotation()
        return self._rotation

    @property
    def width(self) -> float:
        return self.min_bounding_box_mm[1][1]

    @property
    def length(self) -> float:
        return self.min_bounding_box_mm[1][0]

    @property
    def bounding_box(self) -> np.ndarray:
        if self._bounding_box is None:
            self._bounding_box = cv2.boundingRect(self.contour_global)
        return self._bounding_box

    @property
    def bounding_box_mm(self) -> np.ndarray:
        if self._bounding_box_mm is None:
            self._bounding_box_mm = cv2.boundingRect(self.contour_mm('local'))
        return self._bounding_box_mm

    @property
    def min_bounding_box(self) -> np.ndarray:
        if self._min_bounding_box is None:
            self._min_bounding_box = cv2.minAreaRect(self.contour_global)
        return self._min_bounding_box

    @property
    def min_bounding_box_mm(self) -> np.ndarray:
        if self._min_bounding_box_mm is None:
            self._min_bounding_box_mm = cv2.minAreaRect(self.contour_mm('local'))
        return self._min_bounding_box_mm

    @property
    def max_height(self) -> float:
        if self._max_height is None:
            self._max_height = self.height_map[:, :, Z].max()
        return self._max_height

    def contour_idx(self, contour='local') -> tuple:
        if contour == 'local':
            contour = self.contour_local
        elif contour == 'center':
            contour = self.contour_center_local
        else:
            raise
        return tuple(np.hsplit(np.fliplr(contour.reshape(contour.shape[0], 2)), 2))

    def contour_coords(self, contour='local') -> np.ndarray:
        # TODO: повторяет по функционалу contour_mm, удалить последний и решить конфликты
        return self.height_map[self.contour_idx(contour)]

    def contour_z_mean(self, contour='local') -> float:
        return self.contour_coords(contour)[..., Z].mean()

    def contour_z_max(self, contour='local') -> float:
        return np.amax(self.contour_coords(contour)[..., Z])

    def contour_z_min(self, contour='local') -> float:
        return np.amin(self.contour_coords(contour)[..., Z])

    def contour_z_std(self, contour='local') -> float:
        return self.contour_coords(contour)[..., Z].std()

    def find_center_and_rotation(self):
        # center_col, center_row = self.center_local
        # center_z = self.height_map[center_row, center_col, Z]
        # Найти центр и поворот контура по точкам в мм
        # contour_idx = self.contour_idx()  # индексы границы контура
        mean = self.contour_z_mean()
        region = np.where(self.height_map[..., Z] < mean, 0, 255).astype(np.uint8)
        contours = find_contours(region)[0]  # координаты границы где высота выше средней
        if contours:
            contour = contours[0]
        else:
            contour = find_contours(normalize(self.height_map, 255).astype(np.uint8))[0]
        # center_idx = find_center_and_rotation(contour_idx, False)  # индекс центра границы
        # center_idx = tuple(int(round(c)) for c in center_idx)  # округление и перевод в int индекса центра
        center, theta = find_center_and_rotation(
            self.height_map[tuple(np.hsplit(np.fliplr(contour.reshape(contour.shape[0], 2)), 2))][...,
            :Z])  # плоская координата центра граниы
        center_z = mls_height_apprx(self.height_map, center)
        # center, theta = find_center_and_rotation(self.contour_mm('center'))
        center = (*center, center_z)  # простанственные координаты центра (свапнуты из-за opencv)
        rotation = pi / 2 - theta  # перевод в СК принтера
        self._center = center
        self._rotation = rotation

    @property
    def area(self):
        return cv2.moments(self.contour_mm())['m00']

    def copy(self):
        return self.__class__(self.height_map, self.contour_global, self.bounding_box, self.contour_center)

    __copy__ = copy

    def __str__(self) -> str:
        return 'Центр:\n' + \
               f'    X: {self.center[X]: 4.2f} мм\n' + \
               f'    Y: {self.center[Y]: 4.2f} мм\n' + \
               f'    Z: {self.center[Z]: 4.2f} мм\n' + \
               f'Поворот: {self.rotation * 180 / pi:4.2f} градусов\n' + \
               f'Максимальная высота: {self.max_height:4.2f} мм'


def find_cookies(img: Union[np.ndarray, str], height_map: 'np.ndarray') -> (List, np.ndarray):
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
        cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
        col, row, w, h = cv2.boundingRect(contour)
        height_map_masked = height_map.copy()
        height_map_masked[..., Z][mask == 0] = 0
        height_map_fragment = height_map_masked[row:row + h, col:col + w]
        cookie = Cookie(height_map=height_map_fragment, contour_global=contour, bounding_box=(col, row, w, h))
        cookies.append(cookie)
        cv2.circle(result, cookie.center_pixel, 3, (0, 255, 0), -1)
    return cookies, result


def procecc_cookies(cookies: List[Cookie], height_map: np.ndarray, tol: float = 0.05, img: np.ndarray = None) \
        -> (List, np.ndarray):
    pos_img = (height_map[..., Z].copy() / np.amax(height_map[..., Z]) * 255).astype(np.uint8) if img is None else img
    processed = []
    while len(cookies) != 0:
        cookie = cookies.pop()
        # anchor = np.array(cookie.bounding_box[:2])
        mask = np.zeros(height_map.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cookie.contour_global], 0, 255, -1)
        height_map_masked = height_map.copy()
        height_map_masked[..., Z][mask == 0] = 0

        M = cv2.moments(height_map_masked[..., Z])
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
        # while abs(std - p_std) / std > tol and p_std != std:
        #     height_map_masked[..., Z][height_map_masked[..., Z] < mean] = 0
        #     img = (height_map_masked[..., Z] / np.amax(height_map_masked[..., Z]) * 255).astype(np.uint8)
        #     new_cookies, _ = find_cookies(img, height_map_masked)
        #     if len(new_cookies) == 1:
        #         cookie.contour_center = new_cookies[0].contour_global + anchor
        #         p_std = std
        #         c_cnt_z = cookie.contour_coords('center')[..., Z]
        #         c_cnt_z[c_cnt_z == 0] = c_cnt_z.mean()
        #         std = c_cnt_z.std()
        #         mean = c_cnt_z.mean()
        #     elif len(new_cookies) == 0:
        #         break
        #     elif len(new_cookies) > 1:
        #         cookies += [Cookie(new_cookie.height_map, new_cookie.contour_global + anchor) for new_cookie in
        #                     new_cookies if new_cookie.area / cookie.area >= .5]
        #         break
        processed.append(cookie)
        cv2.drawContours(pos_img, [cookie.contour_center], 0, (255, 0, 255), 1)
        cv2.circle(pos_img, (center_x, center_y), 3, (255, 0, 0), -1)
    return processed, pos_img
