#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx


def test_import_main():
    import EasyVision


def test_import_engine():
    import EasyVision.engine


def test_import_engine_objectrecognition():
    from EasyVision.engine import ObjectRecognitionEngine


def test_import_engine_visualodometry():
    from EasyVision.engine import VisualOdometryEngine


def test_import_models():
    import EasyVision.models


def test_import_models_object():
    from EasyVision.models import ObjectModel


def test_import_vision():
    import EasyVision.vision


def test_import_monocular():
    from EasyVision.vision import MonocularVision


def test_import_image():
    from EasyVision.vision import ImagesVision


def test_import_stereopair():
    from EasyVision.vision import StereoPairVision


def test_import_processors():
    import EasyVision.processors


def test_import_featureextractor():
    from EasyVision.processors import FeatureExtraction


def test_import_featureextractor():
    from EasyVision.processors import CalibratedCamera


if __name__ == "__main__":
    import os

    basename = os.path.basename(__file__)
    pytest.main([basename, "-s", "--tb=native"])