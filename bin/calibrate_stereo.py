# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from EasyVision.processorstackbuilder import Args, Builder
from EasyVision.vision import VideoCapture
from EasyVision.processors import CalibratedCamera, CalibratedStereoCamera, StereoCamera, ImageTransform
import json
import cv2


if __name__ == "__main__":
    parser = ArgumentParser(description="Stereo Camera calibration tool")
    parser.add_argument("left", help="Left Camera device ID/folder")
    parser.add_argument("right", help="Left Camera device ID/folder")
    parser.add_argument("-f", "--file", default="stereo_camera.json", help="Output filename of the calibrated camera")
    parser.add_argument("-i", "--size", default="640,480", help="Frame width and height")
    parser.add_argument("-p", "--fps", default="5", help="Frame rate")
    parser.add_argument("-g", "--grid", default="9,7", help="Grid shape of the calibration target")
    parser.add_argument("-N", type=int, default=30, help="Number of samples to gather")
    parser.add_argument("-t", "--test", const=True, default=False, action='store_const',
                        help="Test camera calibration file")
    parser.add_argument("-d", "--disparity", const=True, default=False, action='store_const',
                        help="Calculate Disparity Map")

    args = parser.parse_args()
    grid = tuple(int(i) for i in args.grid.split(','))
    size = tuple(int(i) for i in args.size.split(','))

    try:
        left = int(args.left)
        right = int(args.right)
    except:
        left = args.left
        right = args.right

    camera_model = None
    if args.test:
        with open(args.file) as f:
            camera_model = StereoCamera.fromdict(json.load(f))

    builder = Builder(
        Builder(
            VideoCapture, Args(left),
            ImageTransform, Args(ocl=args.test),
            CalibratedCamera, Args(None if camera_model is None else camera_model.left)
        ),
        Builder(
            VideoCapture, Args(right),
            ImageTransform, Args(ocl=args.test),
            CalibratedCamera, Args(None if camera_model is None else camera_model.right)
        ),
        CalibratedStereoCamera, Args(camera_model, max_samples=args.N, grid_shape=grid,
                                     width=size[0], height=size[1], fps=int(args.fps),
                                     calculate_disparity=args.disparity, display_results=True)
    )

    if args.test:
        with builder.build() as vision:
            for frame in vision:
                cv2.waitKey(1)
    else:
        with builder.build() as vision:
            cam = None
            while not cam:
                cam = vision.calibrate()
                cv2.waitKey(1)

            with open(args.file, "w") as f:
                f.write(json.dumps(cam.todict(), indent=4))
