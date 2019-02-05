#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from tests.common import *


@mark.complex
def test_match_mp_images():
    common_test_match_images('ORB', display=True, mp=True)


@mark.long
def test_match_mp_images_ORB():
    common_test_match_images('ORB', mp=True)


@mark.long
def test_match_mp_images_KAZE():
    common_test_match_images('KAZE', mp=True)


@mark.long
def test_match_mp_images_AKAZE():
    common_test_match_images('AKAZE', mp=True)


@mark.long
def test_match_mp_images_SURF():
    common_test_match_images('SURF', mp=True)


@mark.long
def test_match_mp_images_SIFT():
    common_test_match_images('SIFT', mp=True)
