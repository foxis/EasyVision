#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.base import EasyVisionBase


class Subclass(EasyVisionBase):

    def __init__(self, *args, **kwargs):
        super(Subclass, self).__init__(*args, **kwargs)
        self.frame = 0
        self.frames = 10
        self.release_called = self.setup_called = False

    def setup(self):
        super(Subclass, self).setup()
        self.setup_called = True

    def next(self):
        super(Subclass, self).next()
        if self.is_open:
            return self.capture()
        else:
            raise StopIteration()

    def __len__(self):
        return self.frames

    def capture(self):
        from datetime import datetime
        self.frame += 1
        return (datetime.now, ('an image',))

    def release(self):
        super(Subclass, self).release()
        self.release_called, self.setup_called = True, False

    @property
    def is_open(self):
        return self.frame < self.frames

    @property
    def description(self):
        return "Testing base class"


class SubclassOverload(Subclass):

    def __init__(self, *args, **kwargs):
        self.debug_changed_called = False
        self.debug_changed_called_last = False
        self.debug_changed_called_current = False

        self.display_changed_called = False
        self.display_changed_called_last = False
        self.display_changed_called_current = False
        super(SubclassOverload, self).__init__(*args, **kwargs)

    def debug_changed(self, last, current):
        self.debug_changed_called = True
        self.debug_changed_called_last = last
        self.debug_changed_called_current = current

    def display_results_changed(self, last, current):
        self.display_changed_called = True
        self.display_changed_called_last = last
        self.display_changed_called_current = current


@pytest.mark.main
def test_abstract_base_abstract():
    with raises(TypeError):
        EasyVisionBase()


@pytest.mark.main
def test_implementation():
    tmp = Subclass()
    assert(tmp.name == 'Subclass')


@pytest.mark.main
def test_implementation_context():
    sub = Subclass()
    with sub as s:
        assert(s.setup_called)
    assert(s.release_called)


@pytest.mark.main
def test_iterator():
    with Subclass() as vis:
        count = 0
        for frame in vis:
            count += 1
            if count > 13:
                break
        assert(count == 10)


@pytest.mark.main
def test_setup_release():
    vis = Subclass()
    vis.setup()
    count = 0
    for frame in vis:
        count += 1
        if count > 13:
            break
    assert(count == 10)
    vis.release()


@pytest.mark.main
def test_debug():
    vis = Subclass()
    assert(not vis.debug)
    assert(not vis.display_results)
    vis.debug = True
    assert(not vis.display_results)
    assert(vis.debug)
    vis.debug = False
    assert(not vis.debug)


@pytest.mark.main
def test_display():
    vis = Subclass()
    assert(not vis.display_results)
    assert(not vis.debug)
    vis.display_results = True
    assert(not vis.debug)
    assert(vis.display_results)
    vis.display_results = False
    assert(not vis.display_results)


@pytest.mark.main
def test_debug_init():
    vis = Subclass(debug=True)
    assert(vis.debug)
    assert(not vis.display_results)


@pytest.mark.main
def test_display_init():
    vis = Subclass(display_results=True)
    assert(not vis.debug)
    assert(vis.display_results)


@pytest.mark.main
def test_debug_changed():
    vis = SubclassOverload()
    assert(not vis.debug_changed_called)
    assert(not vis.display_changed_called)
    vis.debug = True
    assert(vis.debug_changed_called)
    assert(not vis.display_changed_called)
    vis.debug_changed_called = False
    vis.debug = True
    assert(not vis.debug_changed_called)
    assert(not vis.display_changed_called)
    vis.debug = False
    assert(vis.debug_changed_called)
    assert(not vis.display_changed_called)


@pytest.mark.main
def test_debug_changed_init():
    vis = SubclassOverload(debug=True)
    assert(vis.name == 'SubclassOverload')
    assert(vis.debug_changed_called)
    assert(not vis.display_changed_called)


@pytest.mark.main
def test_display_changed_init():
    vis = SubclassOverload(display_results=True)
    assert(not vis.debug_changed_called)
    assert(vis.display_changed_called)