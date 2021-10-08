> *!!!IMPORTANT!!!* This repository is being discontinued. All the development is being moved to https://github.com/itohio/EasyVision.

.. image:: https://readthedocs.org/projects/easyvision/badge/?version=latest
    :target: https://easyvision.readthedocs.io/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/foxis/EasyVision.svg?branch=master
    :target: https://travis-ci.org/foxis/EasyVision?branch=master

.. image:: https://codecov.io/gh/FoxIS/EasyVision/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/foxis/EasyVision

.. image:: https://img.shields.io/pypi/v/EasyVision.svg
    :target: https://pypi.python.org/pypi/EasyVision

.. image:: https://img.shields.io/pypi/l/EasyVision.svg
    :target: https://pypi.python.org/pypi/EasyVision

.. image:: https://img.shields.io/pypi/pyversions/EasyVision.svg
    :target: https://pypi.python.org/pypi/EasyVision

.. image:: https://img.shields.io/badge/STAR_Me_on_GitHub!--None.svg?style=social
    :target: https://github.com/foxis/EasyVision

------


.. image:: https://img.shields.io/badge/Link-Document-blue.svg
      :target: https://easyvision.readthedocs.io/index.html

.. image:: https://img.shields.io/badge/Link-API-blue.svg
      :target: https://easyvision.readthedocs.io/py-modindex.html

.. image:: https://img.shields.io/badge/Link-Source_Code-blue.svg
      :target: https://easyvision.readthedocs.io/py-modindex.html

.. image:: https://img.shields.io/badge/Link-Install-blue.svg
      :target: `install`_

.. image:: https://img.shields.io/badge/Link-GitHub-blue.svg
      :target: https://github.com/foxis/EasyVision

.. image:: https://img.shields.io/badge/Link-Submit_Issue-blue.svg
      :target: https://github.com/foxis/EasyVision/issues

.. image:: https://img.shields.io/badge/Link-Request_Feature-blue.svg
      :target: https://github.com/foxis/EasyVision/issues

.. image:: https://img.shields.io/badge/Link-Download-blue.svg
      :target: https://pypi.org/pypi/EasyVision#files


Welcome to ``EasyVision`` Documentation
==============================================================================

Documentation for ``EasyVision`` - a simple framework of computer vision algorithms build on top of OpenCV.
Contains such features as:

    - Calibrated Mono and Stereo camera
    - Feature extraction
    - Color blob extraction
    - Histogram backprojection
    - Background separation
    - Multiprocessing
    - Remote processing using Pyro4
    - Visual Odometry and Mappings
    - Object recognition
    - Bag Of Words (``pip install pyDBoW3``)


.. note::

    Bag Of Words depend on `pyDBoW3 <https://github.com/foxis/pyDBoW3>`_

.. _install:

Install
------------------------------------------------------------------------------

``EasyVision`` is released on PyPI, so all you need is:

.. code-block:: console

    $ pip install EasyVision

To upgrade to latest version:

.. code-block:: console

    $ pip install --upgrade EasyVision
