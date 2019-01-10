#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2
from .common import *
from EasyVision.processorstackbuilder import Builder, Args


class CustomObject(object):

    def __init__(self, a, b):
        self.a, self.b = a, b

    def todict(self):
        return {'a': self.a, 'b': self.b}

    @staticmethod
    def fromdict(d):
        return CustomObject(d['a'], d['b'])

args_truth_simple = {
    'args': (1, 2, 3),
    'kwargs': {'custom': 'color'},
    'objects': {}
}

args_truth_simple_obj = {
    'args': (1, 2, 'object__CustomObject0'),
    'kwargs': {'custom': 'color'},
    'objects': {
        'object__CustomObject0': {'a': 'a var', 'b': 'b var'}
    }
}

args_truth = {
    'args': (1, 2, 3),
    'kwargs': {'custom': 'object__CustomObject1', 'custom1': 'object__CustomObject0'},
    'objects': {
        'object__CustomObject1': {'a': 'a var', 'b': 'b var'},
        'object__CustomObject0': {'a': 'A var', 'b': 'B var'}
    }
}

truth_builder_simple = {
    'args': (
        {
            'args': ('path/to/images',),
            'class': 'VisionSubclass',
            'objects': {},
            'kwargs': {'keyword_argument': True}
        },
        {
            'args': (),
            'class': 'ProcessorA',
            'objects': {},
            'kwargs': {'color': 'color'}
        },
        {
            'args': (),
            'class': 'ProcessorB',
            'objects': {},
            'kwargs': {'camera': 'camera'}
        }
    ),
    'objects': {},
    'kwargs': {
        'display_results': True
    }
}

truth_builder_simple_obj = {
    'args': (
        {
            'args': ('path/to/images',),
            'class': 'VisionSubclass',
            'objects': {},
            'kwargs': {'keyword_argument': True}
        },
        {
            'args': (),
            'class': 'ProcessorA',
            'objects': {},
            'kwargs': {'color': 'color'}
        },
        {
            'args': (),
            'class': 'ProcessorB',
            'objects': {
                'object__CustomObject0': {
                    'a': 'a var',
                    'b': 'b var'
                }
            },
            'kwargs': {'camera': 'object__CustomObject0'}
        }
    ),
    'objects': {},
    'kwargs': {'display_results': True}
}

truth_builder_stereo = {
    'args': (
        {
            'args': (
                {
                    'args': ('path/to/left_images',),
                    'class': 'VisionSubclass',
                    'objects': {},
                    'kwargs': {'keyword_argument': True}
                },
                {
                    'args': (),
                    'class': 'ProcessorA',
                    'objects': {},
                    'kwargs': {'color': 'color left'}
                },
                {
                    'args': (),
                    'class': 'ProcessorB',
                    'objects': {},
                    'kwargs': {'camera': None}
                }
            ),
            'class': 'Builder',
            'kwargs': {'display_results': 1},
            'objects': {},
        },
        {
            'args': (
                {
                    'args': ('path/to/right_images',),
                    'class': 'VisionSubclass',
                    'objects': {},
                    'kwargs': {'keyword_argument': True}
                },
                {
                    'args': (),
                    'class': 'ProcessorA',
                    'objects': {},
                    'kwargs': {'color': 'color right'}
                },
                {
                    'args': (),
                    'class': 'ProcessorB',
                    'objects': {},
                    'kwargs': {'camera': None}
                }
            ),
            'class': 'Builder',
            'kwargs': {'display_results': 2},
            'objects': {},
        },
        {
            'args': (),
            'class': 'ProcessorC',
            'objects': {},
            'kwargs': {'camera': 'camera'}
        }
    ),
    'objects': {},
    'kwargs': {'display_results': True}
}


@mark.main
def test_psb_Args_simple():
    arg = Args(1, 2.5, 3, None, True, False, (1,), [2], {1, 2, 3, 1}, {'a': 'b'}, custom='color', none_type=None, color=cv2.COLOR_BGR2GRAY)
    assert(arg.args == (1, 2.5, 3, None, True, False, (1,), [2], {1, 2, 3}, {'a': 'b'}))
    assert(arg.kwargs == {'custom': 'color', 'none_type': None, 'color': cv2.COLOR_BGR2GRAY})


@mark.main
def test_psb_Args_object():
    arg = Args(1, 2, 3, custom=CustomObject('a var', 'b var'))
    assert(arg.args == (1, 2, 3))
    assert(isinstance(arg.kwargs['custom'], CustomObject))
    assert(arg.kwargs['custom'].a == 'a var')
    assert(arg.kwargs['custom'].b == 'b var')


@mark.main
def test_psb_Args_todict_simple():
    arg = Args(1, 2, 3, custom='color')
    d = arg.todict()
    assert(d == args_truth_simple)


@mark.main
def test_psb_Args_todict_simple_obj():
    arg = Args(1, 2, CustomObject('a var', 'b var'), custom='color')
    d = arg.todict()
    assert(d == args_truth_simple_obj)


@mark.main
def test_psb_Args_todict_kwargs_objects():
    arg = Args(1, 2, 3, custom=CustomObject('a var', 'b var'), custom1=CustomObject('A var', 'B var'))
    d = arg.todict()
    assert(d == args_truth)


@mark.main
def test_psb_Args_fromdict_simple():
    arg = Args.fromdict(args_truth_simple)

    assert(arg.args == (1, 2, 3))
    assert(arg.kwargs == {'custom': 'color'})


@mark.main
def test_psb_Args_fromdict_args_obj():
    arg = Args.fromdict(args_truth_simple_obj, (CustomObject,))

    assert(isinstance(arg.args[2], CustomObject))
    assert(arg.kwargs == {'custom': 'color'})


@mark.main
def test_psb_Args_fromdict_kwargs_objects():
    arg = Args.fromdict(args_truth, (CustomObject,))

    assert(arg.args == (1, 2, 3))
    assert(isinstance(arg.kwargs['custom'], CustomObject))
    assert(arg.kwargs['custom'].a == 'a var')
    assert(arg.kwargs['custom'].b == 'b var')
    assert(isinstance(arg.kwargs['custom1'], CustomObject))
    assert(arg.kwargs['custom1'].a == 'A var')
    assert(arg.kwargs['custom1'].b == 'B var')


@mark.main
def test_psb_Builder_simple():
    builder = Builder(
        VisionSubclass, Args("path/to/images", keyword_argument=True),
        ProcessorA, Args(color='color'),
        ProcessorB, Args(camera='camera'),
        display_results=True
    )

    d = builder.todict()
    assert(d == truth_builder_simple)

    processor = builder.build()

    assert(isinstance(processor, ProcessorB))
    assert(isinstance(processor.source, ProcessorA))
    assert(isinstance(processor.source.source, VisionSubclass))

    assert(processor.display_results)
    assert(processor.source.display_results)
    assert(processor.source.source.display_results)

    assert(processor.camera == 'camera')
    assert(processor.source.color == 'color')
    assert(processor.name == "ProcessorB <- ProcessorA <- path/to/images")

    with processor as vision:
        for frame in vision:
            break


@mark.main
def test_psb_Builder_simple_obj():
    builder = Builder(
        VisionSubclass, Args("path/to/images", keyword_argument=True),
        ProcessorA, Args(color='color'),
        ProcessorB, Args(camera=CustomObject('a var', 'b var')),
        display_results=True
    )

    d = builder.todict()
    assert(d == truth_builder_simple_obj)

    processor = builder.build()

    assert(isinstance(processor, ProcessorB))
    assert(isinstance(processor.source, ProcessorA))
    assert(isinstance(processor.source.source, VisionSubclass))

    assert(processor.display_results)
    assert(processor.source.display_results)
    assert(processor.source.source.display_results)

    assert(isinstance(processor.camera, CustomObject))
    assert(processor.camera.a == 'a var')
    assert(processor.camera.b == 'b var')
    assert(processor.source.color == 'color')
    assert(processor.name == "ProcessorB <- ProcessorA <- path/to/images")

    with processor as vision:
        for frame in vision:
            break


@mark.main
def test_psb_Builder_Builder():
    builder = Builder(
        Builder(
            VisionSubclass, Args("path/to/left_images", keyword_argument=True),
            ProcessorA, Args(color='color left'),
            ProcessorB, Args(camera=None),
            display_results=1
        ),
        Builder(
            VisionSubclass, Args("path/to/right_images", keyword_argument=True),
            ProcessorA, Args(color='color right'),
            ProcessorB, Args(camera=None),
            display_results=2
        ),
        ProcessorC, Args(camera='camera'),
        display_results=True
    )

    d = builder.todict()
    assert(d == truth_builder_stereo)

    processor = builder.build()

    assert(isinstance(processor, ProcessorC))

    assert(isinstance(processor.a, ProcessorB))
    assert(isinstance(processor.a.source, ProcessorA))
    assert(isinstance(processor.a.source.source, VisionSubclass))

    assert(isinstance(processor.b, ProcessorB))
    assert(isinstance(processor.b.source, ProcessorA))
    assert(isinstance(processor.b.source.source, VisionSubclass))

    assert(processor.display_results)
    assert(processor.a.display_results == 1)
    assert(processor.a.source.display_results == 1)
    assert(processor.a.source.source.display_results == 1)
    assert(processor.b.display_results == 2)
    assert(processor.b.source.display_results == 2)
    assert(processor.b.source.source.display_results == 2)

    assert(processor.camera == 'camera')
    assert(processor.name == "ProcessorB <- ProcessorA <- path/to/left_images : ProcessorB <- ProcessorA <- path/to/right_images")

    with processor as vision:
        for frame in vision:
            break


@mark.main
def test_psb_Builder_fromdict_simple():
    builder = Builder.fromdict(truth_builder_simple, (ProcessorA, ProcessorB, VisionSubclass))
    d = builder.todict()
    assert(d == truth_builder_simple)

    processor = builder.build()

    assert(isinstance(processor, ProcessorB))
    assert(isinstance(processor.source, ProcessorA))
    assert(isinstance(processor.source.source, VisionSubclass))

    assert(processor.camera == 'camera')
    assert(processor.source.color == 'color')
    assert(processor.name == "ProcessorB <- ProcessorA <- path/to/images")

    with processor as vision:
        for frame in vision:
            break


@mark.main
def test_psb_Builder_fromdict_simple_obj():
    builder = Builder.fromdict(truth_builder_simple_obj, (ProcessorA, ProcessorB, VisionSubclass, CustomObject))
    d = builder.todict()
    assert(d == truth_builder_simple_obj)

    processor = builder.build()

    assert(isinstance(processor, ProcessorB))
    assert(isinstance(processor.source, ProcessorA))
    assert(isinstance(processor.source.source, VisionSubclass))

    assert(isinstance(processor.camera, CustomObject))
    assert(processor.camera.a == 'a var')
    assert(processor.camera.b == 'b var')
    assert(processor.source.color == 'color')
    assert(processor.name == "ProcessorB <- ProcessorA <- path/to/images")

    with processor as vision:
        for frame in vision:
            break


@mark.main
def test_psb_Builder_Builder_fromdict():
    builder = Builder.fromdict(truth_builder_stereo, (ProcessorA, ProcessorB, ProcessorC, VisionSubclass, CustomObject))

    d = builder.todict()
    assert(d == truth_builder_stereo)

    processor = builder.build()

    assert(isinstance(processor, ProcessorC))

    assert(isinstance(processor.a, ProcessorB))
    assert(isinstance(processor.a.source, ProcessorA))
    assert(isinstance(processor.a.source.source, VisionSubclass))

    assert(isinstance(processor.b, ProcessorB))
    assert(isinstance(processor.b.source, ProcessorA))
    assert(isinstance(processor.b.source.source, VisionSubclass))

    assert(processor.camera == 'camera')
    assert(processor.name == "ProcessorB <- ProcessorA <- path/to/left_images : ProcessorB <- ProcessorA <- path/to/right_images")

    with processor as vision:
        for frame in vision:
            break
