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

Note: You don't have to call ``setup`` for every processor. In fact that will raise an AssertionError, because the last processor will call it's source's
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



Converting image color space
----------------------------


Enabling GPU processing
-----------------------


Multiprocessing
===============


Multiple Consumers
==================


Feature extraction
==================


Blob extraction
===============


Background Separation
=====================


Histogram Backprojection
========================


Calibrated Camera
=================


Calibrated Stereo Camera
========================

