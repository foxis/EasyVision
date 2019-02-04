# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from EasyVision.processorstackbuilder import Builder
from EasyVision.server import Server
from EasyVision.vision import *
from EasyVision.processors import *
import json


if __name__ == "__main__":
    parser = ArgumentParser(description="Remote Processor Stack Server using Pyro4")
    parser.add_argument("name", help="Name of the remote Pyro4 source object")
    parser.add_argument("file", help="Processor Stack builder config file")
    parser.add_argument("-H", "--host", default="localhost", help="Hostname of the server")
    parser.add_argument("-p", "--port", default=0, help="Port of the server")
    parser.add_argument("-r", "--freerun", default=False, help="Whether to go in freerun mode")

    args = parser.parse_args()

    classes = (
        ImagesReader,
        VideoCapture,
        CalibratedCamera,
        CalibratedStereoCamera,
        PinholeCamera,
        StereoCamera,
        ImageTransform,
        FeatureExtraction,
        BackgroundSeparation,
        BlobExtraction
    )
    with open(args.file) as f:
        builder = Builder.fromdict(json.load(f), classes)

    print("Building processor stack...")
    vision = builder.build()

    print("Initializing server...")
    server = Server(args.name, vision, host=args.host, port=args.port, freerun=args.freerun)

    print("Starting server...")
    server.run()
