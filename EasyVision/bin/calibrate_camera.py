# -*- coding: utf-8 -*-
"""Monocular camera calibration tool

Uses rectangular 9x7 calibration pattern for camera intrinsic parameter calibration.
Will output a camera calibration json file, that can be loaded and passed to PinholeCamera object.::

    usage: calibrate_camera.py [-h] [-f FILE] [-g GRID] [-i SIZE] [-p FPS] [-N N]
                               [-t]
                               camera

    Camera calibration tool

    positional arguments:
      camera                Camera device ID/folder

    optional arguments:
      -h, --help            show this help message and exit
      -f FILE, --file FILE  Output filename of the calibrated camera
      -g GRID, --grid GRID  Grid shape of the calibration target
      -i SIZE, --size SIZE  Frame width and height
      -p FPS, --fps FPS     Frame rate
      -N N                  Number of samples to gather
      -t, --test            Test camera calibration file

"""
from argparse import ArgumentParser
from EasyVision.processorstackbuilder import Args, Builder
from EasyVision.vision import VideoCapture
from EasyVision.processors import CalibratedCamera, PinholeCamera
import json
import cv2


def main():
    parser = ArgumentParser(description="Camera calibration tool")
    parser.add_argument("camera", help="Camera device ID/folder")
    parser.add_argument("-f", "--file", default="camera.json", help="Output filename of the calibrated camera")
    parser.add_argument("-g", "--grid", default="9,7", help="Grid shape of the calibration target")
    parser.add_argument("-i", "--size", default="640,480", help="Frame width and height")
    parser.add_argument("-p", "--fps", default="5", help="Frame rate")
    parser.add_argument("-N", type=int, default=30, help="Number of samples to gather")
    parser.add_argument("-t", "--test", const=True, default=False, action='store_const',
                        help="Test camera calibration file")

    args = parser.parse_args()
    grid = tuple(int(i) for i in args.grid.split(','))
    size = tuple(int(i) for i in args.size.split(','))

    try:
        camera = int(args.camera)
    except:
        camera = args.camera

    camera_model = None
    if args.test:
        with open(args.file) as f:
            camera_model = PinholeCamera.fromdict(json.load(f))

    builder = Builder(
        VideoCapture, Args(camera, width=size[0], height=size[1], fps=int(args.fps)),
        CalibratedCamera, Args(camera_model, max_samples=args.N, grid_shape=grid, display_results=True)
    )

    if args.test:
        with builder.build() as vision:
            for frame in vision:
                if cv2.waitKey(1) == 27:
                    break
    else:
        with builder.build() as vision:
            cam = None
            while not cam:
                cam = vision.calibrate()
                cv2.waitKey(1)

            with open(args.file, "w") as f:
                f.write(json.dumps(cam.todict(), indent=4))


if __name__ == "__main__":
    main()