# -*- coding: utf-8 -*-


class EasyVisionError(Exception):
    pass


class NotVisionObject(EasyVisionError):
    pass


class NotModelView(EasyVisionError):
    pass


class TimeoutError(EasyVisionError):
    pass