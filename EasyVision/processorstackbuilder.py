# -*- coding: utf-8 -*-
from EasyVision.vision.base import VisionBase
from EasyVision.processors.base import ProcessorBase
from EasyVision.engine.base import EngineBase


class Args(object):
    """Args class contains arguments for the vision/processor instance creation"""

    MISC_TYPES = (dict, tuple, list, set, frozenset)
    TYPES = (int, long, float, str, unicode, bytearray, bool, complex) + MISC_TYPES

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def todict(self):
        objects = {}

        objects, object_idx, args = Args.convert_args(objects, 0, self.args)
        objects, object_idx, kwargs = Args.convert_kwargs(objects, object_idx, self.kwargs)

        return {
            'args': args,
            'kwargs': kwargs,
            'objects': objects
        }

    @staticmethod
    def convert_args(objects, object_idx, _args):
        args = ()
        for pos, arg in enumerate(_args):
            if isinstance(arg, object) and hasattr(arg, 'todict') and hasattr(arg, 'fromdict') and arg is not None:
                name = 'object__{}{}'.format(arg.__class__.__name__, object_idx)
                objects[name] = arg.todict()
                args += (name,)
                object_idx += 1
            elif not any(isinstance(arg, tp) for tp in Args.TYPES) and arg is not None:
                raise TypeError("Args only support class objects that implement todict/fromdict at position %i" % pos)
            else:
                args += (arg,)
        return objects, object_idx, args

    @staticmethod
    def convert_kwargs(objects, object_idx, _kwargs):
        kwargs = {}
        for arg, value in _kwargs.items():
            if isinstance(value, object) and hasattr(value, 'todict') and hasattr(value, 'fromdict') and value is not None:
                name = 'object__{}{}'.format(value.__class__.__name__, object_idx)
                objects[name] = value.todict()
                kwargs[arg] = name
                object_idx += 1
            elif not any(isinstance(value, tp) for tp in Args.TYPES) and value is not None:
                raise TypeError("Args only support class objects that implement todict/fromdict at kwarg=%s" % arg)
            else:
                kwargs[arg] = value
        return objects, object_idx, kwargs

    @staticmethod
    def fromdict(d, classes=None):
        """Converts a dictionary into Args.
        If dictionary contains custom objects, then classes must be specified.

        :param d: Dictionary that was created using Args.todict()
        :param classes: A list of classes that custom objects can be created from
        :return: New Args object
        """
        classes = Args.convert_classes(classes)
        return Args(*tuple(Args._retrieve_object(classes, d['objects'], arg) for arg in d['args']),
                    **{arg: Args._retrieve_object(classes, d['objects'], value) for arg, value in d['kwargs'].items()})

    @staticmethod
    def convert_classes(classes):
        if isinstance(classes, dict):
            return classes
        return {cls.__name__: cls for cls in classes if hasattr(cls, 'todict') and hasattr(cls, 'fromdict')} if classes else {}

    @staticmethod
    def _retrieve_object(classes, objects, value):
        if isinstance(value, str) and value.startswith('object__'):
            obj = objects[value]
            name = value[8:-1]
            return classes[name].fromdict(obj)
        else:
            return value


class Builder(object):
    """Builder class is a processor stack builder from arguments or dictionary.

    Note: first argument must always be a subclass of VisionBase.
          Even argument must be a subclass of ProcessorBase or Builder object.
          Odd argument must be a class of Args.

    Usage:
        builder = Builder(
            Builder(
                ImagesReader, Args("path/to/left_images", keyword_argument=True),
                ImageTransform, Args(color=cv2.COLOR_BGR2GRAY),
                CalibratedCamera, Args(camera=None),
                FeatureExtraction, Args(feature_type="ORB")
            ),
            Builder(
                ImagesReader, Args("path/to/right_images", keyword_argument=True),
                ImageTransform, Args(color=cv2.COLOR_BGR2GRAY),
                CalibratedCamera, Args(camera=None),
                FeatureExtraction, Args(feature_type="ORB")
            ),
            CalibratedStereoCamera, Args(camera=camera)
            display_results=True
        )

        with builder.build() as vision:
            for frame in vision:
                pass

    fromdict example:
        json = {
            'args': (
                {
                    'class': 'Builder',
                    'args': [
                        {
                            'class': 'ImageReader',
                            'args': ['path/to/images'],
                            'kwargs': {'keyword_argument': True}
                        }
                        {
                            'class': 'ImageTransform',
                            'kwargs': {'color': cv2.COLOR_BGR2GRAY}
                        }
                        {
                            'class': 'CalibratedCamera',
                            'kwargs': {'camera': null}
                        }
                        {
                            'class': 'FeatureExtraction',
                            'kwargs': {'feature_type': "ORB"}
                        }
                    ],
                    'kwargs': {},
                    'objects': {}
                },
                {
                    'class': 'Builder',
                    'args': [
                        {
                            'class': 'ImageReader',
                            'args': ['path/to/images'],
                            'kwargs': {'keyword_argument': True}
                        }
                        {
                            'class': 'ImageTransform',
                            'kwargs': {'color': cv2.COLOR_BGR2GRAY}
                        }
                        {
                            'class': 'CalibratedCamera',
                            'kwargs': {'camera': null}
                        }
                        {
                            'class': 'FeatureExtraction',
                            'kwargs': {'feature_type': "ORB"}
                        }
                    ],
                    'kwargs': {},
                    'objects': {}
                },
                {
                    'class': 'CalibratedStereoCamera',
                    'kwargs': {'camera': 'object__StereoCamera0'},
                    'objects': {
                        'object_StereoCamera0': {...}
                    }
                }
            ),
            'kwargs': {'display_results': True},
            'objects': {}
        ]
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def build(self):
        index = 0
        args = ()
        cls = None
        for pos, arg in enumerate(self.args):
            if isinstance(arg, Builder):
                args += (arg.build(),)
            elif issubclass(arg, VisionBase):
                if index and not (issubclass(arg, ProcessorBase) or issubclass(arg, EngineBase)):
                    raise TypeError("Class at position %i must be either a subclass of ProcessorBase or EngineBase" % pos)
                cls = arg
                index += 1
            elif isinstance(arg, Args) and index % 2:
                if not cls:
                    raise ValueError("Either of Vision/Processor/Engine Base subclass must be provided before Args at position %i" % pos)
                default = {}
                default.update(arg.kwargs)
                default.update(self.kwargs)
                obj = cls(*(args + arg.args), **default)
                args = (obj, )
                index += 1
            else:
                raise ValueError("Invalid arguments at position: %i" % pos)
        assert(len(args) == 1)
        assert(isinstance(args[0], VisionBase) or isinstance(args[1], EngineBase))
        return args[0]

    def todict(self):
        d = {'args': ()}
        cls = None
        for pos, arg in enumerate(self.args):
            if isinstance(arg, Builder):
                args = arg.todict()
                args.update({'class': arg.__class__.__name__})
                d['args'] += (args, )
            elif issubclass(arg, VisionBase) or issubclass(arg, EngineBase):
                cls = arg
            elif isinstance(arg, Args):
                if not cls:
                    raise ValueError("Either VisionBase or ProcessorBase subclass must be provided before Args at position %i" % pos)
                args = arg.todict()
                args.update({'class': cls.__name__})
                d['args'] += (args, )
                cls = None
            else:
                raise TypeError("Unsupported type at position %i" % pos)

            objects, _, kwargs = Args.convert_kwargs({}, 0, self.kwargs)
            d['kwargs'] = kwargs
            d['objects'] = objects
        return d

    @staticmethod
    def fromdict(d, classes=None):
        args = ()
        _classes = Builder.convert_classes(classes)
        arg_classes = Args.convert_classes(classes)
        for pos, arg in enumerate(d['args']):
            if arg['class'] == Builder.__name__:
                args += (Builder.fromdict(arg, _classes),)
            else:
                args += (_classes[arg['class']], Args.fromdict(arg, arg_classes))

        kwargs = {}
        for arg, value in d['kwargs'].items():
            kwargs[arg] = Args._retrieve_object(arg_classes, d['objects'], value)

        return Builder(*args, **kwargs)

    @staticmethod
    def convert_classes(classes):
        if isinstance(classes, dict):
            return classes
        return {cls.__name__: cls for cls in classes} if classes else {}