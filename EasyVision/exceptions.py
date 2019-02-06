# -*- coding: utf-8 -*-
"""This module defines all the base exceptions used throughout EasyVision algorithms.

"""


class EasyVisionError(Exception):
    """General EasyVision exception"""
    pass


class TimeoutError(EasyVisionError):
    """Timeout exception for multithreaded/multiprocessed/RPC code"""
    pass