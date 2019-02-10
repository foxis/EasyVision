Distributed computing
*********************

Distributed computing is useful for processors that take a long time to compute. E.g. FeatureExtraction
and feature matching. There are two methods to achieve distributed computing - one using the same machine multiple
cpu, and second - to use a remote machine on the same network.

Using Multiprocessing
=====================


Using PyroCapture
=================

Since FeatureExtraction doesn't scale properly on the same machine and is usually already parallelized inside
openCV it is possible to relay some lengthy processing to a different machine. This is really useful
for stereo processing, where two machines will capture and process images from left and right camera respectively.
Then CalibratedStereoCamera would capture those processed frames and return as a single frame for further processing
e.g. for Visual Odometry engine.

To achieve that, you need to have two machines on the same network, that run ``EasyVision.bin.server`` instances
for serving captured and processed images. One machine that serves as a NameServer and consumer of those images.

Internally this mechanism uses Pyro4 library.

In case you want to expose other Pyro4 objects, you can pass them to ``Server`` as ``Server(..., objects=my_objects_list)``.
You can expand on ``EasyVision/bin/server.py`` code.