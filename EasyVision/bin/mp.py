# -*- coding: utf-8 -*-
from tests.test_processors_multiprocessing import test_capture_mp_images, test_capture_mp_camera
from tests.test_engine_mp_objectrecognition import test_match_mp_images_ORB
from tests.test_processors_calibratedstereocamera import test_stereo_calibrate, test_stereo_calibrated
from tests.test_processors_mp_calibratedstereocamera import test_stereo_calibrated_mp


if __name__ == "__main__":
    #test_capture_mp_camera()
    #test_match_mp_images_ORB()
    #test_stereo_calibrate()
    test_stereo_calibrated_mp()