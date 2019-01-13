# -*- coding: utf-8 -*-
from EasyVision.vision.base import *
from EasyVision.vision import *
from EasyVision.processors.base import *
from EasyVision.processors import *
from EasyVision.engine import *
import numpy as np
from pytest import approx
import cv2


images_left = ["test_data/left{:02d}.jpg".format(i + 1) for i in range(14) if i != 9]
images_right = ["test_data/right{:02d}.jpg".format(i + 1) for i in range(14) if i != 9]

fp = (535.5289137817749, 535.3112518132406)
cp = (333.9556024187135, 241.22736353333593)
d = [-0.29426670830681295, 0.11409183502487143, 0.0, 0.0, -0.023222122638183858]

M_left =  [[535.5289137817749, 0.0, 333.9556024187135], [0.0, 535.3112518132406, 241.22736353333593], [0.0, 0.0, 1.0]]
d_left =  [[-0.29426670830681295, 0.11409183502487143, 0.0, 0.0, -0.023222122638183858]]
R_left =  [[0.9999334874229316, -0.009219753439929899, 0.0069294210956562], [0.009157022199168523, 0.999917293960849, 0.009030735432780011], [-0.007012109144755233, -0.008966681912493231, 0.9999352123716928]]
P_left =  [[535.3112518132406, 0.0, 325.9803276062012, 0.0], [0.0, 535.3112518132406, 242.73220443725586, 0.0], [0.0, 0.0, 1.0, 0.0]]
M_right =  [[535.5289137817749, 0.0, 332.7230162362328], [0.0, 535.3112518132406, 242.4827739733023], [0.0, 0.0, 1.0]]
d_right =  [[-0.27749259192017467, 0.07216678433491759, 0.0, 0.0, 0.013253845495251491]]
R_right =  [[0.9995094832757231, -0.013756342476967146, 0.02813460295709637], [0.014014659519211728, 0.9998612402293957, -0.009004976707086302], [-0.028006823462464206, 0.009394856496561677, 0.9995635820251514]]
P_right =  [[535.3112518132406, 0.0, 325.9803276062012, -1788.358792054079], [0.0, 535.3112518132406, 242.73220443725586, 0.0], [0.0, 0.0, 1.0, 0.0]]
R =  [[0.9997677227754339, 0.005049397702926733, -0.02095242418578015], [-0.004665553693462927, 0.9998211350625704, 0.018328406666336908], [0.021041223946258476, -0.018226394734858757, 0.9996124576203582]]
T =  [[-3.3391444063671307], [0.04595695668988882], [-0.0939916065445002]]
E =  [[0.0005284677305322451, 0.09313716510823399, 0.04766186280950206], [-0.023710089191732425, -0.06133516502932313, 3.339819698408143], [-0.030367324417407148, -3.3387792054633607, -0.06023828694666684]]
F =  [[-2.553919926621576e-08, -4.502859227625763e-06, -0.00013876836512866469], [1.14630066075936e-06, 2.966548445441988e-06, -0.0875695893199186], [0.0005164591622158981, 0.08722309177066455, 1.0]]
Q =  [[1.0, 0.0, 0.0, -325.9803276062012], [0.0, 1.0, 0.0, -242.73220443725586], [0.0, 0.0, 0.0, 535.3112518132406], [0.0, 0.0, 0.29933101466646467, -0.0]]

left_camera = PinholeCamera((640, 480), M_left, d_left, R_left, P_left)
right_camera = PinholeCamera((640, 480), M_right, d_right, R_right, P_right)

as_dict_left = {
    "size": (640, 480),
    "matrix": M_left,
    "distortion": d_left,
}

as_dict_right = {
    "size": (640, 480),
    "matrix": M_right,
    "distortion": d_right
}

as_dict_stereo = {
    "left": {
        "size": (640, 480),
        "matrix": M_left,
        "distortion": d_left,
        "rectify": R_left,
        "projection": P_left
    },
    "right": {
        "size": (640, 480),
        "matrix": M_right,
        "distortion": d_right,
        "rectify": R_right,
        "projection": P_right
    },
    "R": R,
    "T": T,
    "E": E,
    "F": F,
    "Q": Q
}

# KITTI/note9/tum database
camera_kitti = PinholeCamera.from_parameters((1241, 376), (718.8560, 718.8560), (607.1928, 185.2157), [0.0, 0.0, 0.0, 0.0, 0.0])
camera_kitti_right = PinholeCamera.from_parameters((1241, 376), (718.8560, 718.8560), (607.1928, 185.2157), [0.0, 0.0, 0.0, 0.0, 0.0])
T_kitti =  [[-386.1448], [0], [0]]
camera_note9 = PinholeCamera.from_parameters((1920, 1080), (1920/2, 1080/2), (1920/2, 1080/2), [0.0, 0.0, 0.0, 0.0, 0.0])
camera_tum = PinholeCamera.from_parameters((1280, 1024),
    (1280 * 0.535719308086809, 1024 * 0.669566858850269),
    (1280 * 0.493248545285398, 1024 * 0.500408664348414),
    [0.897966326944875 , 0.0, 0.0, 0.0, 0.0])

NUM_IMAGES = 1591

sequence_tum = 'd:/datasets/vision.in.tum.de/sequence_50/'
dataset_note9 = "d:\datasets\VID_20181217_163202.mp4"


images_obj = [
    "test_data/34838518832_fd00147042_k.jpg",
    "test_data/2732011028_f0f033e678_b.jpg",
    "test_data/4472701625_6b23da9a23_b.jpg",
    "test_data/4472701625_6b23da9a23_b - Copy.jpg",
    "test_data/13669656965_86a0146858_k.jpg",
    "test_data/21503937795_a9f41f428a_k.jpg"
]
image_obj1 = "test_data/4472701625_6b23da9a23_b.jpg"
image_obj1_crop = "test_data/4472701625_6b23da9a23_b_crop1.jpg"
image_obj1_mask = "test_data/4472701625_6b23da9a23_b - mask.jpg"
image_obj2_crop = "test_data/4472701625_6b23da9a23_b_crop2.jpg"


class MyException(Exception): pass


class VisionSubclass(VisionBase):

    def __init__(self, name="", keyword_argument=None, *args, **kwargs):
        super(VisionSubclass, self).__init__(*args, **kwargs)
        self.frame = 0
        self.frames = 10
        self._name = name
        self._camera_called = False
        self._test_remote_get = 'success'
        self._autoexposure = None
        self._autofocus = None
        self._autowhitebalance = None
        self._autogain = None
        self._exposure = None
        self._focus = None
        self._whitebalance = None
        self.keyword_argument = keyword_argument

    def capture(self):
        from datetime import datetime
        self.frame += 1
        return Frame(datetime.now(), self.frame - 1, (Image(self, 'an image'),))

    def setup(self):
        super(VisionSubclass, self).setup()

    def release(self):
        super(VisionSubclass, self).release()

    def camera_(self):
        self._camera_called = True
        return True

    @property
    def camera_called(self):
        return self._camera_called

    @property
    def is_open(self):
        return self.frame < self.frames

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        pass

    @property
    def path(self):
        pass

    @property
    def frame_size(self):
        pass

    @property
    def fps(self):
        pass

    @property
    def frame_count(self):
        return self.frames

    @property
    def devices(self):
        """
        :return: [{name:, description:, path:, etc:}]
        """
        pass

    @property
    def autoexposure(self):
        return self._autoexposure

    @property
    def autofocus(self):
        return self._autofocus

    @property
    def autowhitebalance(self):
        return self._autowhitebalance

    @property
    def autogain(self):
        return self._autogain

    @property
    def exposure(self):
        return self._exposure

    @property
    def focus(self):
        return self._focus

    @property
    def whitebalance(self):
        return self._whitebalance

    @property
    def gain(self):
        return self._gain

    @autoexposure.setter
    def autoexposure(self, value):
        self._autoexposure = value

    @autofocus.setter
    def autofocus(self, value):
        self._autofocus = value

    @autowhitebalance.setter
    def autowhitebalance(self, value):
        self._autowhitebalance = value

    @autogain.setter
    def autogain(self, value):
        self._autogain = value

    @exposure.setter
    def exposure(self, value):
        self._exposure = value

    @focus.setter
    def focus(self, value):
        self._focus = value

    @whitebalance.setter
    def whitebalance(self, value):
        self._whitebalance = value

    @gain.setter
    def gain(self, value):
        self._gain = value

    @property
    def test_remote_getter_only(self):
        return 'some_constant'

    @property
    def test_remote_get(self):
        print 'remote get', self, self._test_remote_get
        return self._test_remote_get

    @test_remote_get.setter
    def test_remote_get(self, value):
        print 'remote set', self, value
        self._test_remote_get = value

    def test_remote_call(self, a, b, kwarg_test=0):
        print 'remote call', self, self._test_remote_get, a, b, kwarg_test
        return (self._test_remote_get, a, b, kwarg_test)

    def test_remote_exception(self, a, b, kwarg_test=0):
        raise MyException()

    def display_results_changed(self, last, current):
        pass


class ProcessorA(ProcessorBase):

    def __init__(self, vision, camera=None, color=None, *args, **kwargs):
        super(ProcessorA, self).__init__(vision, *args, **kwargs)
        self.camera = camera
        self.color = color

    @property
    def description(self):
        return "Simple processor"

    def process(self, image):
        new_image = image.image.upper()
        return image._replace(source=self, image=new_image)

    def display_results_changed(self, last, current):
        pass


class ProcessorB(ProcessorBase):

    def __init__(self, vision, camera=None, color=None, *args, **kwargs):
        super(ProcessorB, self).__init__(vision, *args, **kwargs)
        self.camera = camera
        self.color = color

    @property
    def description(self):
        return "Simple processor 2"

    def process(self, image):
        new_image = image.image.title()
        return image._replace(source=self, image=new_image)

    def display_results_changed(self, last, current):
        pass


class ProcessorC(ProcessorBase):

    def __init__(self, a, b, camera=None, color=None, *args, **kwargs):
        self.a = a
        self.b = b
        self.camera = camera
        a.camera = 'left camera %s' % camera
        b.camera = 'right camera %s' % camera
        self.color = color
        super(ProcessorC, self).__init__(None, *args, **kwargs)

    @property
    def description(self):
        return "Simple processor 2"

    def setup(self):
        self.a.setup()
        self.b.setup()
        super(ProcessorBase, self).setup()

    def release(self):
        self.a.release()
        self.b.release()
        super(ProcessorBase, self).release()

    def display_results_changed(self, last, current):
        pass

    def capture(self):
        a = self.a.capture()
        b = self.a.capture()
        if a is None or b is None:
            return None

        return a._replace(images=a.images + b.images)

    def process(self, image):
        raise NotImplemented()

    @property
    def name(self):
        return "{} : {}".format(self.a.name, self.b.name)


def assert_camera(camera):
    assert(camera.size == (640, 480))
    assert(camera.focal_point[0] == approx(fp[0]))
    assert(camera.center[0] == approx(cp[0]))
    assert(camera.distortion[0][0] == approx(d[0]))
    assert(isinstance(camera.matrix, np.ndarray))
    assert(isinstance(camera.distortion, np.ndarray))


def assert_stereo_camera(camera):
    assert(isinstance(camera.left, PinholeCamera))
    assert(isinstance(camera.right, PinholeCamera))
    assert(isinstance(camera.R, np.ndarray))
    assert(isinstance(camera.T, np.ndarray))
    assert(isinstance(camera.E, np.ndarray))
    assert(isinstance(camera.F, np.ndarray))
    assert(isinstance(camera.Q, np.ndarray))


def common_test_match_images(feature_type, display=False, mp=False):
    vision = ImagesReader(images_obj)
    _extractor = FeatureExtraction(vision, feature_type, display_results=display)
    extractor = MultiProcessing(_extractor, freerun=False, display_results=display, debug=display) if mp else _extractor
    with ObjectRecognitionEngine(extractor, feature_type, display_results=display) as engine:
        frame_count = 0

        assert(engine.enroll("obj1", ImagesReader.load_image(image_obj1, image_obj1_mask), add=True, display_results=display) is not None)
        assert(engine.enroll("obj2", ImagesReader.load_image(image_obj2_crop), add=True, display_results=display) is not None)
        assert(len(engine.models) == 2)

        for frame, matches in engine:
            frame_count += 1
            assert(isinstance(frame, Frame))
            assert(frame.images[0].image is not None)

            if frame.index == 2:
                if display and len(matches) != 2:
                    cv2.waitKey(0)
                assert(len(matches) == 2)
                assert(matches[0].model.name == 'obj1' or matches[0].model.name == 'obj2')
                assert(matches[1].model.name == 'obj1' or matches[1].model.name == 'obj2')
            elif frame.index == 3:
                if display and len(matches) != 3:
                    cv2.waitKey(0)
                assert(len(matches) == 3)
            else:
                assert(len(matches) == 0)
        if display:
            cv2.waitKey(0)

        assert(frame_count == len(images_obj))


def common_test_visual_odometry_kitti(feature_type, mp=False, ocl=True, debug=False, color=cv2.COLOR_BGR2GRAY, odometry_class=VisualOdometry2DEngine,
                                      pose="00"):
    traj = np.zeros((600, 600, 3), dtype=np.uint8)


    images_kitti = ['d:/datasets/data_odometry_gray/dataset/sequences/{}/image_0/{}.png'.format(pose, str(i).zfill(6)) for i in xrange(NUM_IMAGES)]
    gt_path_kitti = "d:/datasets/data_odometry_gray/dataset/poses/{}.txt".format(pose)

    with open(gt_path_kitti) as f:
        ground_truth = [[float(i) for i in line.split()] for line in f.readlines()]

    error = 0
    cam = CalibratedCamera(ImageTransform(ImagesReader(images_kitti), ocl=ocl, color=color, enabled=True), camera_kitti, display_results=False, enabled=False)
    cam = MultiProcessing(cam, freerun=False) if mp else cam
    with odometry_class(cam, display_results=True, debug=debug, feature_type=feature_type) as engine:
        for img_id, _ in enumerate(images_kitti):
            true_x = ground_truth[img_id][3]
            true_y = ground_truth[img_id][7]
            true_z = ground_truth[img_id][11]

            if img_id > 0:
                scale = np.sqrt((true_x - x_prev) ** 2 + (true_y - y_prev) ** 2 + (true_z - z_prev) ** 2)
                x_prev, y_prev, z_prev = true_x, true_y, true_z
            else:
                scale = 1.0
                x_prev, y_prev, z_prev = true_x, true_y, true_z

            frame, pose = engine.compute(absolute_scale=scale)
            if pose:
                t = pose.translation

                error += np.sqrt((true_x - t[0]) ** 2 + 0 * (true_y - t[1]) ** 2 + (true_z - t[2]) ** 2)

                draw_x, draw_y = int(t[0]) + 290, int(t[2]) + 90
                dtrue_x, dtrue_y = int(true_x) + 290, int(true_z) + 90

                cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0))
                cv2.circle(traj, (dtrue_x, dtrue_y), 1, (0, 0, 255))
                cv2.rectangle(traj, (0, 0), (600, 60), (0, 0, 0), -1)
                text = "pose: x=%2fm y=%2fm z=%2fm" % (t[0], t[1], t[2])
                cv2.putText(traj, text, (20, 11), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "true: x=%2fm y=%2fm z=%2fm" % (true_x, true_y, true_z)
                cv2.putText(traj, text, (20, 22), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "scale: %2f" % scale
                cv2.putText(traj, text, (20, 33), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "cumulative error: %2f " % error
                cv2.putText(traj, text, (20, 44), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

            cv2.imshow('Trajectory', traj)
            cv2.waitKey(1)

    cv2.waitKey(0)
