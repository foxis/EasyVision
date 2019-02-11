#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.base import EasyVisionBase


class Subclass(EasyVisionBase):

    def __init__(self, *args, **kwargs):
        super(Subclass, self).__init__(*args, **kwargs)
        self._frame = 0
        self._frames = 10
        self._release_called = self._setup_called = False

    def setup(self):
        super(Subclass, self).setup()
        self._setup_called = True

    def next(self):
        super(Subclass, self).next()
        if self.is_open:
            return self.capture()
        else:
            raise StopIteration()

    def __len__(self):
        return self._frames

    def capture(self):
        from datetime import datetime
        self._frame += 1
        return (datetime.now, ('an image',))

    def release(self):
        super(Subclass, self).release()
        self._release_called, self._setup_called = True, False

    @property
    def is_open(self):
        return self._frame < self._frames

    @property
    def description(self):
        return "Testing base class"


class SubclassOverload(Subclass):

    def __init__(self, *args, **kwargs):
        self._debug_changed_called = False
        self._debug_changed_called_last = False
        self._debug_changed_called_current = False

        self._display_changed_called = False
        self._display_changed_called_last = False
        self._display_changed_called_current = False
        super(SubclassOverload, self).__init__(*args, **kwargs)

    def debug_changed(self, last, current):
        self._debug_changed_called = True
        self._debug_changed_called_last = last
        self._debug_changed_called_current = current

    def display_results_changed(self, last, current):
        self._display_changed_called = True
        self._display_changed_called_last = last
        self._display_changed_called_current = current


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
        assert(s._setup_called)
    assert(s._release_called)


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
    assert(not vis._debug_changed_called)
    assert(not vis._display_changed_called)
    vis.debug = True
    assert(vis._debug_changed_called)
    assert(not vis._display_changed_called)
    vis._debug_changed_called = False
    vis.debug = True
    assert(not vis._debug_changed_called)
    assert(not vis._display_changed_called)
    vis.debug = False
    assert(vis._debug_changed_called)
    assert(not vis._display_changed_called)


@pytest.mark.main
def test_debug_changed_init():
    vis = SubclassOverload(debug=True)
    assert(vis.name == 'SubclassOverload')
    assert(vis._debug_changed_called)
    assert(not vis._display_changed_called)


@pytest.mark.main
def test_display_changed_init():
    vis = SubclassOverload(display_results=True)
    assert(not vis._debug_changed_called)
    assert(vis._display_changed_called)