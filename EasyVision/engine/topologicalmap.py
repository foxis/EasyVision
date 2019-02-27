# -*- coding: utf-8 -*-
"""Implements simple TopologicalMap on top of OccupancyGridMap using BOW
"""

from EasyVision.engine.occupancygridmap import OccupancyGridMap
from EasyVision.engine.base import Pose
import numpy as np
import cv2
from math import sin, cos, acos, asin
import os
import pyDBoW3 as bow
from collections import namedtuple


ORB_VOCABULARY_PATH = os.path.join("", "EasyVision", "engine", "orbvoc.dbow3")


class Node(namedtuple("Node", "pose links")):
    """Namedtuple class declaring graph nodes.

        **pose**: a pose for this node

        **links**: references to other nodes this one has connection to.

    .. note::
        pose may have rotation/translation set to None. In that case path planning
        will calculate relative rotation/translation of the current pose to target pose.

    """

    def todict(self):
        return {
            "pose": self.pose.todict(),
            "links": self.links
        }

    def fromdict(self, d):
        return Node(Pose.fromdict(d['pose']), d['links'])


class TopologicalMap(OccupancyGridMap):
    """Class that implements Topological Map on top of OccupancyGridMap and uses pyDBoW3 library
    for topology building.

    On every update call, updates both the topology(visibility graph) and occupancy grid.
    """

    def __init__(self, _map, vocabulary, *args, graph=None, **kwargs):
        """Instance initialization.

        :param _map: Accepts either a tuple of (map_width, map_height) or a np.float32 two dimensional array
        :param vocabulary: a path to BOW vocabulary in DBoW3 format.
        :param graph: a list of Node elements
        """
        if graph is None:
            graph = []

        self._database = None
        self._results = None
        self._voc_path = vocabulary
        self._graph = graph

        print("myself init")

        super(TopologicalMap, self).__init__(_map, *args, **kwargs)

    def setup(self):
        print("myself setup")
        self._database = bow.Database()
        self._database.loadVocabulary(self._voc_path, False, -1)

        print("myself setup pregraph")

        for pose in self._graph:
            if pose.features is None:
                raise AttributeError("All poses must contain features")
            # TODO except for the first pose that initiates the origin of the path
            self._database.add(pose.features.descriptors)
            print("myself setup add pose")

        super(TopologicalMap, self).setup()

    def release(self):
        print("myself release")
        super(TopologicalMap, self).release()

    @property
    def description(self):
        return "Topological map using OccupancyGridMap and BOW"

    @property
    def graph(self):
        return self._graph

    def update(self, pose, **kwargs):
        print("myself update")
        updated_pose = super(TopologicalMap, self).update(pose, **kwargs)

        if pose.features is None:
            raise AttributeError("All poses must contain features")

        results = self._database.query(pose.features.descriptors, 5, -1)
        self._results = [(result.Score, result.Id) for result in results]
        print("myself update query", self._results)

        if not results or max(score for score, id in self._results) < 0.01:
            self._database.add(pose.features.descriptors)
            self._graph.append(Node(pose, tuple(result.Id for result in results)))

        # TOOD:
        # query database
        # if some_criteria:
        #   database.add
        #   super().update
        #   add a node
        #   update node links based on query results

        return updated_pose

    def draw(self, path=None):
        disp = super(TopologicalMap, self).draw(path, display=False)

        matches = tuple(id for _, id in self._results) if self._results else ()

        for id, node in enumerate(self._graph):
            draw_x, draw_y = int(node.pose.translation[0] * self._scale), int(node.pose.translation[2] * self._scale)
            c = (0, 255, 0) if id in matches else (255, 0, 0)
            cv2.circle(disp, (draw_x, draw_y), 1, c)

        cv2.imshow(self.name, disp)

        # TODO:
        # draw the map
        # draw nodes
        # draw matching nodes with different color

    def plan(self, target, radius, **kwargs):
        """

        there are several modes of planning:

            - plan using grid map
            - plan using graph

            - plan when features are available
            - plan when features are not available
            - plan when rotate/translate is available
            - plan when rotate/translate is not available

        also planning may depend whether graph contains rotate/translate or not.

        use case::

            setup the graph so that we only have features. in that case calculate
            rotate/translate based on relative rotate/translate

        :param target:
        :param radius:
        :param kwargs:
        :return:
        """
        pass
