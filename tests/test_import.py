#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx


def test_import_main():
    import EasyVision
    pass


def test_import_engine():
    import EasyVision.engine
    pass


def test_import_engine_objectrecognition():
    from EasyVision.engine import ObjectRecognitionEngine
    pass


def test_import_models():
    import EasyVision.models
    pass


def test_import_models_object():
    from EasyVision.models import ObjectModel
    pass


def test_import_vision():
    import EasyVision.vision
    pass


def test_import_monocular():
    from EasyVision.vision import MonocularVision
    pass


def test_import_stereopair():
    from EasyVision.vision import StereoPairVision
    pass


if __name__ == "__main__":
    import os

    basename = os.path.basename(__file__)
    pytest.main([basename, "-s", "--tb=native"])