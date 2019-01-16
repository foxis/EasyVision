#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx


@pytest.mark.main
def test_import_main():
    import EasyVision


@pytest.mark.main
def test_import_engine():
    import EasyVision.engine


@pytest.mark.main
def test_import_engine_objectrecognition():
    from EasyVision.engine import ObjectRecognitionEngine


@pytest.mark.main
def test_import_engine_visualodometry2d():
    from EasyVision.engine import VisualOdometry2DEngine


@pytest.mark.main
def test_import_engine_visualodometry3d2d():
    from EasyVision.engine import VisualOdometry3D2DEngine


@pytest.mark.main
def test_import_engine_visualodometry_stereo():
    from EasyVision.engine import VisualOdometryStereoEngine


@pytest.mark.main
def test_import_models():
    import EasyVision.models


@pytest.mark.main
def test_import_models_object():
    from EasyVision.models import ObjectModel


@pytest.mark.main
def test_import_vision():
    import EasyVision.vision


@pytest.mark.main
def test_import_monocular():
    from EasyVision.vision import VideoCapture


@pytest.mark.main
def test_import_image():
    from EasyVision.vision import ImagesReader


@pytest.mark.main
def test_import_multiprocessing():
    from EasyVision.processors import MultiProcessing


@pytest.mark.main
def test_import_multiconsumers():
    from EasyVision.processors import MultiConsumers


@pytest.mark.main
def test_import_processors():
    import EasyVision.processors


@pytest.mark.main
def test_import_featureextractor():
    from EasyVision.processors import FeatureExtraction


@pytest.mark.main
def test_import_histogrambackprojection():
    from EasyVision.processors import HistogramBackprojection


@pytest.mark.main
def test_import_blobextractor():
    from EasyVision.processors import BlobExtraction


@pytest.mark.main
def test_import_featureextractor():
    from EasyVision.processors import CalibratedCamera


@pytest.mark.main
def test_import_topologicalslam():
    from EasyVision.engine import TopologicalSLAMEngine


@pytest.mark.main
def test_import_bowvocabulary():
    from EasyVision.engine import BOWVocabularyBuilderEngine


if __name__ == "__main__":
    import os

    basename = os.path.basename(__file__)
    pytest.main([basename, "-s", "--tb=native"])