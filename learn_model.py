# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from EasyVision.processorstackbuilder import Args, Builder
from EasyVision.vision import VideoCapture
from EasyVision.processors import CalibratedCamera, PinholeCamera, FeatureExtraction, BackgroundSeparation
from EasyVision.engine import ObjectRecognitionEngine
from EasyVision.models import ObjectModel
import json
import cv2


if __name__ == "__main__":
    parser = ArgumentParser(description="Camera calibration tool")
    parser.add_argument("device", help="Camera device ID/folder")
    parser.add_argument("-f", "--file", default="model.json", help="Output filename of the learned model")
    parser.add_argument("-c", "--camera", default="camera.json", help="Calibrated PinholeCamera file")
    parser.add_argument("-e", "--feature_type", default="ORB", help="Feature Type (e.g. ORB/FREAK/SIFT)")
    parser.add_argument("-i", "--size", default="640,480", help="Frame width and height")
    parser.add_argument("-p", "--fps", default="5", help="Frame rate")
    parser.add_argument("-t", "--test", const=True, default=False, action='store_const',
                        help="Test learned model")

    args = parser.parse_args()
    size = tuple(int(i) for i in args.size.split(','))

    try:
        camera = int(args.device)
    except:
        camera = args.device

    camera_model = None
    object_model = None
    if args.test:
        with open(args.file) as f:
            pass
            object_model = ObjectModel.fromdict(json.load(f))
    try:
        with open(args.camera) as f:
            camera_model = PinholeCamera.fromdict(json.load(f))
    except:
        pass

    if args.test:
        builder = Builder(
            VideoCapture, Args(camera, width=size[0], height=size[1], fps=int(args.fps)),
            CalibratedCamera, Args(camera_model, enabled=camera_model is not None),
            FeatureExtraction, Args(feature_type=args.feature_type),
            ObjectRecognitionEngine, Args(feature_type=args.feature_type, display_results=True)
        )
        with builder.build() as engine:
            for frame in engine:
                if cv2.waitKey(1) == 27:
                    break
    else:
        builder = Builder(
            VideoCapture, Args(camera, width=size[0], height=size[1], fps=int(args.fps)),
            CalibratedCamera, Args(camera_model, enabled=camera_model is not None),
            BackgroundSeparation, Args(algorithm="MOG"),
            FeatureExtraction, Args(feature_type=args.feature_type, display_results=True),
            ObjectRecognitionEngine, Args(feature_type=args.feature_type, display_results=True)
        )
        with builder.build() as vision:
            for frame, obj in vision:

                model = vision.enroll("object", frame, add=True, display_results=True)
                if model:
                    object_model = model
                ch = cv2.waitKey(1)
                if ch == 27:
                    break
                elif ch == 32:
                    vision.vision.background_num = 0

            with open(args.file, "w") as f:
                f.write(json.dumps(object_model.todict(), indent=4))
