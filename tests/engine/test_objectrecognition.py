#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from tests.common import *


@mark.complex
def test_match_images_ORB_complex():
    common_test_match_images('SIFT', display=True)


@mark.long
def test_match_images_ORB():
    common_test_match_images('ORB')


@mark.long
def test_match_images_KAZE():
    common_test_match_images('KAZE')


@mark.long
def test_match_images_AKAZE():
    common_test_match_images('AKAZE')


@mark.long
def test_match_images_SURF():
    common_test_match_images('SURF')


@mark.long
def test_match_images_SIFT():
    common_test_match_images('SIFT')
