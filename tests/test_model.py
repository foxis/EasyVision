#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.models.base import *


class Subclass(ModelBase):

    def __init__(self, name, views, *args, **kwargs):
        super(Subclass, self).__init__(name, views)

    def compute(self, frame, views):
        pass

    def release(self):
        pass

    @property
    def description(self):
        pass


def test_abstract():
    with raises(TypeError):
        ModelBase()


def test_implementation():
    model = Subclass('empty model', [])
    assert(not len(model))
    assert(model.name == 'empty model')


def test_model_view():
    view = ModelView('image', 'mask', 'features', 'feature type')
    assert(view.image == 'image')
    assert(view.outline == 'mask')
    assert(view.features == 'features')
    assert(view.feature_type == 'feature type')


def test_add_model_view():
    model = Subclass('empty model', [])
    model.update(ModelView('image', 'outline', 'features', 'feature type'))
    assert(len(model))

def test_add_model():
    model = Subclass('model', [ModelView('image', 'outline', 'features', 'feature type')])
    model2 = Subclass('new model', [ModelView('image1', 'outline1', 'features1', 'feature type')])
    model.update(model2)
    assert(len(model) == 2)
