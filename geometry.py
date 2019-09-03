import numpy as np
from numpy import cos, sin, tan, pi


class Vector3:
    def __init__(self, x=0, y=0, z=0):
        """

        :param float x:
        :param float y:
        :param float z:
        """
        self.x = x
        self.y = y
        self.z = z
        self.vector = np.array([x, y, z])

    def __str__(self):
        return f'({self.x:4.2f} {self.y:4.2f} {self.z:4.2f})'

    def __repr__(self):
        return f'({self.x:4.2f} {self.y:4.2f} {self.z:4.2f})'

    @classmethod
    def fromNumPy(cls, nparr):
        """

        :param np.ndarray nparr:
        :return Vector3:
        """
        if nparr.shape == (3,):
            [x, y, z] = nparr
            return cls(x, y, z)
        else:
            raise Exception(f'Shape does not match. Expected (3,) got {nparr.shape} instead.')

    @classmethod
    def fromList(cls, list):
        """

        :param list list:
        :return Vector3:
        """
        if len(list) == 3:
            [x, y, z] = list
            return cls(x, y, z)
        else:
            raise Exception(f'Length does not match. Expected 3 got {len(list)} instead.')

    def __add__(self, other):
        """

        :param Vector3 other:
        :return Vector3:
        """
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Vector3(x, y, z)

    def __sub__(self, other):
        """

        :param Vector3 other:
        :return Vector3:
        """
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Vector3(x, y, z)

    def __abs__(self):
        return np.linalg.norm(self.vector)

    def __mul__(self, other):
        """

        :param float other:
        :return Vector3:
        """
        return Vector3(self.x * other, self.y * other, self.z * other)

    def dot(self, other, reverse=False):
        """

        :param Vector3 other:
        :return float:
        """
        return np.dot(self.vector, other.vector)

    def cross(self, other):
        """

        :param Vector3 other:
        :return Vector3:
        """
        return Vector3.fromNumPy(np.cross(self.vector, other.vector))

    def rotate2d(self, theta, rotation_center=None):
        """

        :param float theta:
        :param Vector3 rotation_center:
        :return:
        """
        sin_t = sin(theta)
        cos_t = cos(theta)
        R = np.array([[cos_t, -sin_t, 0],
                      [sin_t, cos_t, 0],
                      [0, 0, 1]])
        if rotation_center is None:
            rotation_center = Vector3(0, 0, 0)
        v = self - rotation_center
        v = Vector3.fromNumPy(np.dot(R, v.vector))
        v += rotation_center
        return v
