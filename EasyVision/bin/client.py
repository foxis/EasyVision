from EasyVision.vision import PyroCapture
from EasyVision.vision import Frame
from EasyVision.processors import PinholeCamera
from datetime import datetime
import cv2
import time
from argparse import ArgumentParser

W, H = 640, 480

camera = PinholeCamera.from_parameters(
    (W, H),
    (W/2, H/2),
    (W/2, H/2),
    [0.0, 0.0, 0.0, 0.0, 0.0]
)


if __name__ == "__main__":
    parser = ArgumentParser(description="Simple Remote Processor Client using Pyro4")
    parser.add_argument("name", help="Name of the remote Pyro4 source object")
    parser.add_argument("-N", type=int, default=300, help="Number of frames to capture")

    args = parser.parse_args()

    with PyroCapture(args.name) as vis:
        print(vis.name)
        print(vis.fps)
        print(vis.frame_size)
        print(vis.frame_count)
        print(vis.camera)

        vis.camera = camera

        print(vis.camera)

        name = vis.name

        now = datetime.now()
        for i in range(30):
            vis._proxy.echo('dat' * 640*480)
        print("echo calls ps with 640*480*3 bytes", 30 / (datetime.now() - now).total_seconds())

        print("letting to freerun for 10s")
        time.sleep(10)
        print('remote fps', vis._proxy.fps())

        now = datetime.now()
        for i, frame in enumerate(vis):
            assert(isinstance(frame, Frame))
            print('.',)
            if i > args.N:
                break
            cv2.imshow(name, frame.images[0].image)
            cv2.waitKey(1)

        print('capture fps', args.N / (datetime.now() - now).total_seconds())
        print('remote fps', vis._proxy.fps())
