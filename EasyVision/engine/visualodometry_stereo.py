# -*- coding: utf-8 -*-
from .base import *
from EasyVision.processors.base import *
from EasyVision.processors import FeatureExtraction, StereoCamera, CalibratedStereoCamera, FeatureMatchingMixin
import cv2
import numpy as np


class VisualOdometryStereoEngine(FeatureMatchingMixin, OdometryBase):

    def __init__(self, vision, feature_type=None, pose=None,
                 num_features=None, nlevels=None,
                 ratio=None, distance_thresh=None, reproj_thresh=None, reproj_error=None,
                 min_dZ=None, max_dZ=None,
                 *args, **kwargs):
        feature_extractor_provided = False
        if not isinstance(vision, ProcessorBase) and not isinstance(vision, CalibratedStereoCamera):
            raise TypeError("Vision must be either CalibratedStereoCamera or ProcessorBase")

        if isinstance(vision, ProcessorBase):
            if vision.get_source('CalibratedStereoCamera') is None:
                raise TypeError("Vision must contain CalibratedStereoCamera")

            if vision.get_source('FeatureExtraction') is not None:
                fe = vision.get_source('FeatureExtraction')
                assert(fe[0].feature_type == fe[1].feature_type)

                feature_type = fe[0].feature_type
                feature_extractor_provided = True
            elif not feature_type:
                raise TypeError("Feature type must be provided")
        else:
            raise TypeError("Vision must contain CalibratedStereoCamera")

        defaults = {}

        if feature_type == 'ORB':
            defaults['nfeatures'] = num_features if num_features is not None else 2000
            #defaults['scoreType'] = cv2.ORB_FAST_SCORE
            defaults['nlevels'] = nlevels if nlevels is not None else 16
            defaults.update(kwargs)
            defaults.pop('enabled', None)
            defaults.pop('debug', None)
            defaults.pop('display_results', None)

        self._feature_type = feature_type
        _vision = FeatureExtraction(vision, feature_type=feature_type, **defaults) if not feature_extractor_provided else vision

        self._camera = _vision.camera
        assert(isinstance(self.camera, StereoCamera))
        self._last_frame = None
        self._pose = pose
        self._last_pose = None
        self._last_3dfeatures = None

        self._ratio = 0.7
        self._distance_thresh = 100
        self._min_matches = 10
        self._reproj_thresh = 0.3
        self._reproj_error = .5
        self._dZ = 300
        self._max_dZ = 500

        if feature_type == 'ORB':
            self._distance_thresh = 100
            self._reproj_error = 2
        elif feature_type == 'FREAK':
            self._distance_thresh = 90
            self._reproj_error = .2
        elif feature_type == 'SIFT':
            self._distance_thresh = 200
        elif feature_type == 'BRISK':
            self._distance_thresh = 80

        if ratio is not None:
            self._ratio = ratio
        if distance_thresh is not None:
            self._distance_thresh = distance_thresh
        if reproj_thresh is not None:
            self._reproj_thresh = reproj_thresh
        if reproj_error is not None:
            self._reproj_error = reproj_error
        if min_dZ is not None:
            self._dZ = min_dZ
        if max_dZ is not None:
            self._max_dZ = max_dZ

        super(VisualOdometryStereoEngine, self).__init__(_vision, *args, **kwargs)

    def compute(self):
        frame = self.vision.capture()
        if not frame:
            return None

        stereo_features = self._calculate_3d(frame.images[0].features, frame.images[1].features)

        if self._last_3dfeatures is not None:
            matches = self._match_stereo(self._last_3dfeatures, stereo_features)
            self._last_3dfeatures = stereo_features

            if matches is None or not matches:
                return frame, self._pose

            _, points_3d, new_points_2d, new_points_3d = matches

            r, t = None, None
            use_rt = False
            if self._last_pose:
                R, t = self._last_pose
                r, _ = cv2.Rodrigues(R)
                r *= -1
                t *= -1 / 1.4
                use_rt = True

            ret, r, t, inliers = cv2.solvePnPRansac(points_3d, new_points_2d, self._camera.left.matrix, None,
                                                    r, t, use_rt, reprojectionError=self._reproj_error)
            R, _ = cv2.Rodrigues(r * -1)
            t *= -1.4

            dZ = sum(i[0] ** 2 for i in t) ** .5

            if ret and dZ < self._dZ:
                if self._pose:
                    self._pose = self._pose._replace(translation=self._pose.translation + self._pose.rotation.dot(t), rotation=R.dot(self._pose.rotation))
                else:
                    self._pose = Pose(R, t)
                self._last_pose = Pose(R, t)

            if self.debug:
                if dZ > 3000:
                    print 'fail'
                    for a, b in zip(points_3d, new_points_3d):
                        print a, b

                cv2.rectangle(self.features, (0, 0), (600, 600), (0, 0, 0), -1)

                scale = max(max(p[0], p[2]) for p in points_3d)
                yscale = max(p[1] for p in points_3d)
                scale = 50000
                yscale = 1000
                #for p in point_3d:
                #    c = max(0, 255 - int(200 * p[1] / yscale))
                #    cv2.circle(self.features, (300 + int(300 * p[0] / scale), 600 - int(600 * p[2] / scale)), 1, (0, 0, c))

                for a, b in zip(points_3d, new_points_3d):
                    cv2.line(self.features,
                        (300 + int(300 * a[0] / scale), 600 - int(600 * a[2] / scale)),
                        (300 + int(300 * b[0] / scale), 600 - int(600 * b[2] / scale)),
                        (0, 255, 0))

                text = "Number of last 3d features: %i" % len(self._last_3dfeatures[2])
                cv2.putText(self.features, text, (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "Number of new 3d features: %i" % len(stereo_features[2])
                cv2.putText(self.features, text, (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                text = "Number of matching features: %i" % len(new_points_2d)
                cv2.putText(self.features, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                text = "Number of inliers: %i" % len(inliers)
                cv2.putText(self.features, text, (20, 55), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                text = "dZ min: %f" % MIN
                cv2.putText(self.features, text, (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "dZ max: %f" % MAX
                cv2.putText(self.features, text, (20, 85), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                text = "dZ pos: %f" % (dZ / 3)
                cv2.putText(self.features, text, (20, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                img = frame.images[0].image.get() if isinstance(frame.images[0].image, cv2.UMat) else frame.images[0].image

                #left, right = stereo_features[:2]
                #for l, r in zip(left, right):
                #    cv2.line(img, (int(l[0]), int(l[1])), (int(r[0]), int(r[1])), (0, 0, 255))

                for l, n in zip(points_2d, new_points_2d):
                    cv2.line(img, (int(l[0]), int(l[1])), (int(n[0]), int(n[1])), (255, 0, 0))

                cv2.imshow("Left feature matches to right", img)

        if self.debug:
            cv2.imshow("3D points", self.features)

        self._last_3dfeatures = stereo_features
        return frame, self._pose

    def _calculate_3d(self, featuresA, featuresB):
        """ Finds stereo correspondances and triangulates the points.
        will return corresponding left/right points and triangulated points
        :return: (left, right, 3d points) or None
        """
        kpsA, descriptorsA = featuresA
        kpsB, descriptorsB = featuresB
        matches = self._match_features(descriptorsA, descriptorsB, self._feature_type,
                        self._ratio, self._distance_thresh, self._min_matches)

        if matches is None or not matches:
            return None

        umat_descriptors = isinstance(descriptorsA, cv2.UMat)
        if umat_descriptors:
            descriptorsA = descriptorsA.get()
            descriptorsB = descriptorsB.get()

        kpsA = [kpsA[m.queryIdx] for m in matches]
        kpsB = [kpsB[m.trainIdx] for m in matches]
        mask = [abs(a.pt[1] - b.pt[1]) < 2 and a.pt[0] > b.pt[0] for a, b in zip(kpsA, kpsB)]

        dA = np.array([descriptorsA[m.queryIdx] for M, m in zip(mask, matches) if M], dtype=descriptorsA.dtype)
        dB = np.array([descriptorsB[m.trainIdx] for M, m in zip(mask, matches) if M], dtype=descriptorsB.dtype)

        kpsA = [p for M, p in zip(mask, kpsA) if M]
        kpsB = [p for M, p in zip(mask, kpsB) if M]

        left = np.float32([kp.pt for kp in kpsA])
        right = np.float32([kp.pt for kp in kpsB])

        E, mask = cv2.findEssentialMat(left, right, focal=self._camera.left.focal_point[0], pp=self._camera.left.center,
                                       method=cv2.RANSAC, prob=0.999, threshold=self._reproj_thresh)

        kpsA = [kp for m, kp in zip(mask, kpsA) if m]
        kpsB = [kp for m, kp in zip(mask, kpsB) if m]
        dA = np.array([d for m, d in zip(mask, dA) if m])
        dB = np.array([d for m, d in zip(mask, dB) if m])

        left = np.float32([kp.pt for kp in kpsA])
        right = np.float32([kp.pt for kp in kpsB])

        if self._camera.left.projection is None or self._camera.left.projection is None:
            P1 = np.dot(self._camera.left.matrix, np.hstack((np.eye(3, 3), np.zeros((3, 1)))))
            P2 = np.dot(self._camera.right.matrix, np.hstack((self._camera.R, self._camera.T)))
        else:
            P1 = self._camera.left.projection
            P2 = self._camera.right.projection

        point_4d_hom = cv2.triangulatePoints(P1, P2, np.expand_dims(left, axis=1), np.expand_dims(right, axis=1))
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d[:3, :].T

        if umat_descriptors:
            dA = cv2.UMat(dA)
            dB = cv2.UMat(dB)

        return left, right, point_3d, dA

    def _match_stereo(self, last_features, new_features):
        """Matches Last frame features with new frame features.
        Filters matched features based on triangulated points.

        :return: (last2d, last3d, new2d, new3d) or None
        """
        matches = self._match_features(last_features[3], new_features[3],
                self._feature_type, self._ratio, self._distance_thresh / 3, self._min_matches)

        if matches is None:
            return None

        last_points_3d = [last_features[2][m.queryIdx] for m in matches]
        new_points_3d = [new_features[2][m.trainIdx] for m in matches]

        dZ = 3 * sum(i[0] ** 2 for i in self._last_pose.translation) ** .5 if self._last_pose else self._dZ
        dZ = min(self._max_dZ, max(self._dZ, dZ))
        self._dZ = dZ
        dZ = 300
        mask = [abs(a[2] - b[2]) < dZ for a, b in zip(last_points_3d, new_points_3d)]

        if self.debug:
            self.MIN = min(a[2] - b[2] for a, b in zip(last_points_3d, new_points_3d))
            self.MAX = max(a[2] - b[2] for a, b in zip(last_points_3d, new_points_3d))

        if sum(mask) < self._min_matches:
            return None

        new_points_3d = np.float32([p for M, p in zip(mask, new_points_3d) if M])
        last_points_3d = np.float32([p for M, p in zip(mask, last_points_3d) if M])
        new_points_2d = np.float32([new_features[0][m.trainIdx] for M, m in zip(mask, matches) if M])
        last_points_2d = np.float32([last_features[0][m.queryIdx] for M, m in zip(mask, matches) if M])

        return last_points_2d, last_points_3d, new_points_2d, new_points_3d

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def camera(self):
        return self._camera

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, value):
        if not isinstance(value, Pose) and value is not None:
            raise TypeError("Pose must be of type Pose")
        self._pose = value

    @property
    def relative_pose(self):
        return self._last_pose

    @property
    def camera_orientation(self):
        pass

    @camera_orientation.setter
    def camera_orientation(self, value):
        pass

    @property
    def description(self):
        return "Stereo Visual Odometry"

    @property
    def capabilities(self):
        return EngineCapabilities(
                (ProcessorBase, CalibratedStereoCamera, FeatureExtraction),
                (Frame, Pose),
                {'feature_type': ('FREAK', 'SURF', 'SIFT', 'ORB', 'KAZE', 'AKAZE')}
            )

    def debug_changed(self, last, current):
        if current:
            cv2.namedWindow("3D points")
            self.features = np.zeros((600, 600, 3), dtype=np.uint8)