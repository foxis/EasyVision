EasyVision Utilities
********************

EasyVision provides several tools to ease certain commonly used tasks, such as camera calibration,
learning of model features for object recognition, or learning color histogram as well as remote server.

Camera Calibration
==================

Checker board 9x7 calibration pattern is being used for mono and stereo camera calibration.
You can find an image with the pattern in openCV images folder.

Mono Camera: calibrate_camera
-----------------------------

Usage::

    # python -m EasyVision.bin.calibrate_camera -h
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


Stereo Camera: calibrate_stereo
-------------------------------

Usage::

    # python -m EasyVision.bin.calibrate_stereo -h
    usage: calibrate_stereo.py [-h] [-f FILE] [-i SIZE] [-p FPS] [-g GRID] [-N N]
                               [-t] [-d]
                               left right

    Stereo Camera calibration tool

    positional arguments:
      left                  Left Camera device ID/folder
      right                 Left Camera device ID/folder

    optional arguments:
      -h, --help            show this help message and exit
      -f FILE, --file FILE  Output filename of the calibrated camera
      -i SIZE, --size SIZE  Frame width and height
      -p FPS, --fps FPS     Frame rate
      -g GRID, --grid GRID  Grid shape of the calibration target
      -N N                  Number of samples to gather
      -t, --test            Test camera calibration file
      -d, --disparity       Calculate Disparity Map

Histogram learning tool: learn_histogram
========================================

This utility will start a GUI where you can select a rectangle from which the histogram should be calculated.
After you press Escape it will write histogram Json file of the learnt histogram.

You can read this histogram into python array and then convert it into numpy array in order to
use with ``BlobExtraction`` or ``HistogramBackprojection`` processors.

Usage::

    # python EasyVision.bin.learn_histogram -h
    usage: learn_histogram.py [-h] [-f FILE] [-i SIZE] [-t] camera

    Color histogram learning tool

    positional arguments:
      camera                Camera device ID/folder

    optional arguments:
      -h, --help            show this help message and exit
      -f FILE, --file FILE  Output filename of the learned histogram
      -i SIZE, --size SIZE  Frame width and height
      -t, --test            Test histogram


Object learning tool: learn_model
=================================

This utility will start a GUI, where you can see what model views are encoded, and what it is matched against.

Usage::

    # python -m EasyVision.bin.learn_model -h
    usage: learn_model.py [-h] [-f FILE] [-c CAMERA] [-H HAND] [-e FEATURE_TYPE]
                          [-i SIZE] [-p FPS] [-t]
                          device

    Camera calibration tool

    positional arguments:
      device                Camera device ID/folder

    optional arguments:
      -h, --help            show this help message and exit
      -f FILE, --file FILE  Output filename of the learned model
      -c CAMERA, --camera CAMERA
                            Calibrated PinholeCamera file
      -H HAND, --hand HAND  Color histogram of the hand
      -e FEATURE_TYPE, --feature_type FEATURE_TYPE
                            Feature Type (e.g. ORB/FREAK/SIFT)
      -i SIZE, --size SIZE  Frame width and height
      -p FPS, --fps FPS     Frame rate
      -t, --test            Test learned model


Remote processing server: server
================================

Usage::

    # python -m EasyVision.bin.server -h
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

    # python -m EasyVision.bin.server 'LeftCamera' left-stack.json -H 0.0.0.0
