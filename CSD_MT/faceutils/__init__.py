#!/usr/bin/python
# -*- encoding: utf-8 -*-
from . import faceplusplus as fpp
def get_dlib():
    """
    惰性加载 dlibutils。只有当你真的需要 align_mode='dlib' 时再调用：
        f = faceutils.get_dlib()
        faces = f.detect(img)
    平时 align_mode='skip' 不会 import dlib。
    """
    from . import dlibutils as dlib  # 如果你把 dlibutils 删了，这个函数就不要被调用
    return dlib

