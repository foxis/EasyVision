Image processing
****************

``EasyVision.processing`` submodule contains various image processing algorithms. Image processing objects unlike capturing objects support stacking.
Meaning, that you can perform Image color space conversion, calibrated camera undistort and feature extraction in that order.
Since each processor class requires either a capture or processor object as it's first input you can do the following:

.. code-block:: python

    capturing = VideoCapturing(0)
    p1 = ImageTransform(capturing, color=cv2.COLOR_BGR2GRAY)
    p2 = CalibratedCamera(p1, camera=camera)
    p3 = FeatureExtraction(p2)

    with p3 as vision:
        for frame in vision:
            # do something with the frame

Alternatively you can build a processor stack using ``Processor Stack Builder`` like this:

.. code-block:: python

    builder = Builder(
        VideoCapturing, Args(0),
        ImageTransform, Args(color=cv2.COLOR_BGR2GRAY),
        CalibratedCamera, Args(camera=camera),
        FeatureExtraction, Args()
    )

    with builder.build() as vision:
        for frame in vision:
            # do something with the frame

For more detailed information on this way of building processor stack, refer to ``Processor Stack Builder`` page.

.. note::

    You don't have to call ``setup`` for every processor. In fact that will raise an AssertionError, because the last processor will call it's source's
    ``setup`` method, so that each processor and capturing object will be properly initialized. Also note, that in this example ``setup`` is called implicitly
    using context manager pattern.

Processing mask
===============

Since a Frame object may contain variable amout of images it is useful to instruct a specific processor to process only certain images.
This is done using ``processor_mask`` parameter specified at the creation of processing object.
Processing mask is a simple string containing 0 and 1 like so "01" - this means, that only second image in the frame should be processed.

Duplicating v.s. replacing images
=================================

Sometimes it is useful to add a processed image to the frame instead of replacing it. This is done by passing ``append=True`` during creation of a processor.

Image transform
===============

``ImageTransform`` processor provides simple image processing.

Converting image color space
----------------------------

By providing ``color=cv2.COLOR_*`` value during processor instantiation, ``ImageTransform`` will
call ``cvtColor`` method on the source images.

Enabling GPU processing
-----------------------

By providing ``ocl=True`` during processor instantiation images will be transferred to a GPU by
using cv2.UMat. If ImageTransform encounters an image already in the gpu and ``ocl=False`` was set,
then the image will be brought back from GPU. Also it will transfer feature descriptors from GPU memory.

Multiprocessing
===============

Sometimes it is useful to run several processors in parallel. ``MultiProcessing`` processor provides
that functionality.

.. note::

    Some of OpenCV algorithms do not provide any meaningful performance benefit by using multiprocessing.
    For example FeatureExtraction does not run in parallel.
    Another thing to keep in mind is that internally MultiProcessing uses pipes to transfer frames
    between processes which incurrs some performance penalty.

Multiple Consumers
==================

Sometimes you want generate a branched processor stack. You can do that with ``MultiConsumer`` processor.

This is how you could use this processor:

.. code-block:: python

    c = VideoCapture(0)
    mc = MultiConsumer(c)
    p1 = FeatureExtraction(mc, feature_type='ORB')
    p2 = BlobExtraction(mc, histogram)

    with p1 as features:
        with p2 as blobs:
            for f, b in zip(features, blobs):
                # do something with feature frame and blob frame

Although you can achieve the same with the following:

.. code-block:: python

    c = VideoCapture(0)
    p1 = FeatureExtraction(c, feature_type='ORB')
    p2 = BlobExtraction(p1, histogram, append=True)

    with p2 as features_and_blobs:
        for frame in features_and_blobs:
            # frame.images[0].features is features
            # frame.images[1].features is blobs

.. note::
    ``MultiConsumer`` will capture a new frame only when all the consumers have called ``capture`` on this
    processor. Number of consumers is determined by the number of ``setup`` calls were performed.
    number of ``release`` calls should match the number of ``setup`` calls.

Feature extraction
==================

``FeatureExtraction`` processor will use OpenCV to detect and extract features from images.

``FeatureMatchingMixin`` is available to help feature matching for classes that want this functionality.

This processor will populate ``features`` and ``feature_type`` fields in the processed image.

.. note::
    SURF and SIFT features are under respective patents and so are removed from standard OpenCV build.
    You can access these features by using ``opencv-contrib-python==3.4.2.16`` package or build it yourself.

.. note::
    This processor does not scale properly when using ``MultiProcessing`` processor.

Blob extraction
===============

``BlobExtraction`` processor will use OpenCV color blob extraction given a color histogram.

``BlobMatchingMixin`` is available to help blob matching for classes that want this functionality.

You can use ``HistogramBackprojection`` processor to calculate a histogram for blob extraction.
You can also use ``learn_histogram`` utility to learn a histogram of an object.

This processor will populate ``features`` and ``feature_type`` fields in the processed image.

Background Separation
=====================

This processor is useful for scenese, where background is not changed(stationary camera) and all
you need is to capture only moving or dynamic objects in the scene.

This processor will calculate a mask, that is useful to filter out background from moving objects.

.. note::
    This processor must capture at least N frames to learn background model.

Histogram Backprojection
========================

This processor will calculate a mask for situations where you need to filter out some color.
You can check the source code of ``learn_model`` utility how this is used to filter out a hand
holding an object.

Calibrated Camera
=================

This processor uses OpenCV Pinhole Camera model to calibrate and undistort calibrated camera images.

You can check the code of ``calibrate`` utility on how the calibration process is being used.

Calibrated Stereo Camera
========================

This processor uses two ``CalibratedCamera`` as it's source to calibrate or undistort/rectify
image pairs to produce stereo images. Can also calculate disparity map.

``CalibratedStereoCamera`` will set respective ``CalibratedCamera`` object's ``camera`` properties inside
``__init__`` method.

.. note::
    Currently there is no functionality to synchronise frames from two cameras.
