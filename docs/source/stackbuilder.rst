Processor Stack Builder
***********************

Building from code
==================

Building processor stack from code is rather simple. All you need to provide is a processor/capturer class, an instance of Args like so:

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

You can even provide other builders like so:

.. code-block:: python

    builder = Builder(
        Builder(
            VideoCapture, Args(0),
            ImageTransform, Args(ocl=True),
            CalibratedCamera, Args(None)
        ),
        Builder(
            VideoCapture, Args(1),
            ImageTransform, Args(ocl=True),
            CalibratedCamera, Args(None)
        ),
        CalibratedStereoCamera, Args(camera_model)
    )

Building from JSON
==================

