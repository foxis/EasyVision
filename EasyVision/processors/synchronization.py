# -*- coding: utf-8 -*-
"""Allows to synchronize capture using a callback.
"""

from .base import *


class Synchronize(ProcessorBase):
    """ Synchronization processor that allows to synchronize two processing streams,
    that possibly run on different machines.

    Basically it accepts a callable which will be called before capturing a frame
    and it should block until all processing streams are in sync.
    """

    def __init__(self, vision, sync, *args, **kwargs):
        """Synchronize instance initialization

        :param vision: capturing source object
        :param sync: a blocking callable responsible for synchronization
        """

        self._sync = sync

        super(Synchronize, self).__init__(vision, *args, **kwargs)

    @property
    def description(self):
        return "Processing stream synchronizing processor"

    def process(self, image):
        return self.source.process(image)

    def capture(self):
        super(ProcessorBase, self).capture()
        if self.enabled:
            self._sync()
        return self.source.capture()
