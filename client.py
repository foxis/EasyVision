from EasyVision.vision import PyroCapture
from EasyVision.vision import Frame
from EasyVision.processors import PinholeCamera
from datetime import datetime
import cv2

camera = PinholeCamera.from_parameters(
    (640, 480),
    (640/2, 480/2),
    (640/2, 480/2),
    [0.0, 0.0, 0.0, 0.0, 0.0]
)

with PyroCapture('test') as vis:
    print vis.name
    print vis.fps
    print vis.frame_size
    print vis.frame_count
    print vis.camera

    vis.camera = camera

    print vis.camera

    name = vis.name

    now = datetime.now()
    N = 300
    for i, frame in enumerate(vis):
        assert(isinstance(frame, Frame))
        print '.',
        if i > N:
            break
        cv2.imshow(name, frame.images[0].image)
        cv2.waitKey(1)

    print N / (datetime.now() - now).total_seconds()
