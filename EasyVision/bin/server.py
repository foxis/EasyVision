# -*- coding: utf-8 -*-
"""Remote Processor Stack Server using Pyro4. Pyro4 NameServer must be running.::

    usage: server.py [-h] [-H HOST] [-p PORT] [-l] name file

    Remote Processor Stack Server using Pyro4 Processor Stack Builder json

    positional arguments:
      name                  Name of the remote Pyro4 source object
      file                  Processor Stack builder Json file

    optional arguments:
      -h, --help            show this help message and exit
      -H HOST, --host HOST  Hostname of the server (default: localhost)
      -p PORT, --port PORT  Port of the server (default: 0)
      -l, --lazy            Specifies whether to do lazy capturing, e.g. on demand
                            (default: false)


Processor Stack Builder json example:
=====================================
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
                "kwargs": {"ocl":  false}
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
                "kwargs": {"camera": "object__PinholeCamera0"}
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
"""
from argparse import ArgumentParser
from EasyVision.processorstackbuilder import Builder
from EasyVision.server import Server
from EasyVision.vision import *
from EasyVision.processors import *
import json


def main():
    parser = ArgumentParser(description="""Remote Processor Stack Server using Pyro4
    Processor Stack Builder json example:

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
                "kwargs": {"ocl":  false}
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
                "kwargs": {"camera": "object__PinholeCamera0"}
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
    """)
    parser.add_argument("name", help="Name of the remote Pyro4 source object")
    parser.add_argument("file", help="Processor Stack builder Json file")
    parser.add_argument("-H", "--host", default="localhost", help="Hostname of the server (default: localhost)")
    parser.add_argument("-n", "--nameserver", default=None, help="hostname of the name server (default: empty)")
    parser.add_argument("-p", "--port", default=0, help="Port of the server (default: 0)")
    parser.add_argument("-l", "--lazy", const=True, default=False, action='store_const',
                        help="Specifies whether to do lazy capturing, e.g. on demand (default: false)")

    args = parser.parse_args()

    classes = (
        ImagesReader,
        VideoCapture,
        PyroCapture,
        CalibratedCamera,
        CalibratedStereoCamera,
        PinholeCamera,
        StereoCamera,
        ImageTransform,
        FeatureExtraction,
        BackgroundSeparation,
        HistogramBackprojection,
        BlobExtraction,
        MultiThreading,
        MultiProcessing
    )
    with open(args.file) as f:
        builder = Builder.fromdict(json.load(f), classes)

    print("Building processor stack...")
    vision = builder.build()

    print("Initializing server...")
    server = Server(args.name, vision, host=args.host, port=args.port, nameserver=args.nameserver, freerun=not args.lazy)

    print("Starting server...")
    server.run()


if __name__ == "__main__":
    main()
