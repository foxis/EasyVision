# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from EasyVision.vision import VideoReader
from EasyVision.processors import CalibratedCamera, CalibratedStereoCamera
import json


if __name__ == "__main__":
    parser = ArgumentParser(description="Stereo Camera calibration tool")
    parser.add_argument("left", help="Left Camera device ID/folder")
    parser.add_argument("right", help="Left Camera device ID/folder")
    parser.add_argument("-f", "--file", default="camera.json", required=True, help="Output filename of the calibrated camera")
    parser.add_argument("-N", type=int, default=30, help="Number of samples to gather")

    args = parser.parse_args()

    left = CalibratedCamera(VideoReader(args.left), None)
    right = CalibratedCamera(VideoReader(args.right), None)
    with CalibratedStereoCamera(left, right, None, max_samples=args.N, display_results=True) as vision:
        cam = None
        while not cam:
            cam = vision.calibrate()

        with open(args.file) as f:
            f.write(json.dumps(cam.todict(), indent=4))
