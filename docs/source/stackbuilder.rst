Processor Stack Builder
***********************

Building from code
==================

Building processor stack from code is rather simple. All you need to provide is a processor/capturer class, an instance of Args like so:

.. code-block:: python

    builder = Builder(
        VideoCapturing, Args(0),
        ImageTransform, Args(color=cv2.COLOR_BGR2GRAY),
        CalibratedCamera, Args(camera=camera),
        FeatureExtraction, Args()
    )

    with builder.build() as vision:
        for frame in vision:
            # do something with the frame

You can even provide other builders like so:

.. code-block:: python

    builder = Builder(
        Builder(
            VideoCapture, Args(0),
            ImageTransform, Args(ocl=True),
            CalibratedCamera, Args(None)
        ),
        Builder(
            VideoCapture, Args(1),
            ImageTransform, Args(ocl=True),
            CalibratedCamera, Args(None)
        ),
        CalibratedStereoCamera, Args(camera_model)
    )

By having a builder object, you can also convert it into a dictionary object with ``todict`` method. This
is convenient if you want to serialize a builder into e.g. JSON.

.. note::

    Only objects, that expose ``todict`` method can be passed to Args in order to convert a builder into a dictionary.
    Standard python types(such as tuple, list, int, string, etc.) are not required to have this method, obviously.

Building from JSON
==================

Builder object provides ``todict`` and ``fromdict`` methods to convert the processor stack to/from a dictionary
which is useful for creating a builder from e.g. JSON file.

Dictionary format
-----------------

A root level of the dictionary contains such keys as ``args``, ``objects`` and ``kwargs``.

    ``args`` will contain a list of arguments that were provided to the builder.

    ``objects`` will contain all the serialized objects that were provided to ``Args``.

    ``kwargs`` are the keyword arguments that were supplied to the builder.

``args`` is a list of dictionaries, that contains these fields:

    ``args`` is a list of positional arguments provided to ``Args``.

    ``kwargs`` is a dictionary of keyword arguments provided to ``Args``

    ``class`` is a name of the class that was provided to the builder before ``Args``

    ``objects`` is a dictionary containing all the serialized objects provided to ``Args``

The structure looks like this:

.. code-block:: python

    class MyObject(object):
        def todict(self):
            ...
        @classmethod
        def fromdict(d):
            ...
            return MyObject(...)

    my_obj = MyObject(...)

    builder = Builder(
        ClassA, Args(0, 1, kwarg1="kw1"),
        ClassB, Args(3, my_obj),
        kwarg2="kw2"
    )

    builder.todict()
    {
        "args": [
            {
                "class": ClassA,
                "args": [0, 1],
                "kwargs": {"kwarg1": "kw1"},
                "objects": {}
            },
            {
                "class": ClassB,
                "args": [2, 3, "object__MyObject0"],
                "kwargs": {},
                "objects": {
                    "object__MyObject0": {
                        ...
                    }
                }
            }
        ],
        "objects": {},
        "kwargs": {"kwarg2": "kw2"}
    }

Objects that were provided to ``Args`` are serialized by calling their ``todict`` method and are replaced with
"object__{ObjectClassName}{index of the object serialized}".

In the example json file below, ``PinholeCamera`` object was passed to ``CalibratedCamera`` processor as a
keyword argument, and thus it was replaced with "object__PinholeCamera0" and it's serialized version was
inserted into ``objects`` dictionary.

Simple JSON usecase example
---------------------------

Here is an example JSON file:

.. code-block:: json

    {
        "args": [
            {
                "args": [0],
                "class": "VideoCapture",
                "objects": {},
                "kwargs": {"width":  1280, "height":  720}
            },
            {
                "args": [],
                "class": "ImageTransform",
                "objects": {},
                "kwargs": {"ocl":  true}
            },
            {
                "args": [],
                "class": "CalibratedCamera",
                "objects": {
                    "object__PinholeCamera0": {
                        "rectify": null,
                        "distortion": [
                            [
                                0.07507829590903714,
                                0.2133670120228787,
                                0.004960489645345226,
                                -0.0019449662761104394,
                                -1.0317011493764785
                            ]
                        ],
                        "projection": null,
                        "matrix": [
                            [
                                732.8937676878295,
                                0.0,
                                311.31379638926603
                            ],
                            [
                                0.0,
                                728.1072411106162,
                                261.6539111360498
                            ],
                            [
                                0.0,
                                0.0,
                                1.0
                            ]
                        ],
                        "size": [
                            640,
                            480
                        ]
                    }
                },
                "kwargs": {"camera": null}
            },
            {
                "args": [],
                "class": "FeatureExtraction",
                "objects": {},
                "kwargs": {"feature_type": "ORB", "enabled":  true}
            }
        ],
        "objects": {},
        "kwargs": {}
    }

And this code will build the same processor stack as in the first example from a json file:

.. code-block:: python

    with open('builder.json') as f:
        stack = json.load(f)

    classes = (
        VideoCapture,
        ImageTransform,
        FeatureExtraction,
        PinholeCamera
    )

    builder = Builder.fromdict(stack, classes)
    with builder.build() as vision:
        for frame in vision:
            # do something with the frame

.. note::
    You must supply classes that are provided in the dictionary. TypeError will be raised if
    a class specified in the dictionary is not found in the supplied class list.
