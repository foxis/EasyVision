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
                 min_dZ=None, max_dZ=None, max_dY=None, max_dX=None,
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
            defaults['nfeatures'] = num_features if num_features is not None else 3000
            #defaults['scoreType'] = cv2.ORB_FAST_SCORE
            defaults['nlevels'] = nlevels if nlevels is not None else 4
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
        self._reproj_error = 5
        self._dZ = 600
        self._max_dZ = self._dZ * 2
        self._dY = 2
        self._dX = 300

        if feature_type == 'ORB':
            self._distance_thresh = 120
            self._reproj_error = 5
        elif feature_type == 'FREAK':
            self._distance_thresh = 200
            self._reproj_error = 5
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
        if max_dX is not None:
            self._dX = max_dX
        if max_dY is not None:
            self._dY = max_dY

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
                self._last_frame = frame
                print 'no matches'
                return frame, self._pose

            last_points_2d, last_points_3d, new_points_2d, new_points_3d, last_points_2d_right, new_points_2d_right = matches

            _r, _t = None, None
            use_rt = False
            if self._last_pose:
                R, _t = self._last_pose
                _r, _ = cv2.Rodrigues(R)
                _r *= -1
                _t *= -1
                use_rt = True

            for iteration in range(10):
                ret, r, t, inliers = cv2.solvePnPRansac(last_points_3d, new_points_2d, self._camera.left.matrix, None,
                                                        _r, _t, use_rt, reprojectionError=self._reproj_error, confidence=.999 - .1 * iteration, iterationsCount=100)
                projected_2d, _ = cv2.projectPoints(last_points_3d, r, t, self._camera.left.matrix, None)

                reproj_error_inliers = sum(p.dot(p) ** .5 for i, p in enumerate(a - b[0] for a, b in zip(new_points_2d, projected_2d)) if i in inliers) / len(inliers)
                reproj_error = sum(p.dot(p) ** .5 for p in (a - b[0] for a, b in zip(new_points_2d, projected_2d))) / len(new_points_2d)
                if reproj_error_inliers < self._reproj_error:
                    break
                else:
                    print 'failed to find inliers', reproj_error_inliers, '<', self._reproj_error
            else:
                ret = False

            R, _ = cv2.Rodrigues(r * -1)
            t *= -1

            dZ = sum(i[0] ** 2 for i in t) ** .5

            if ret:
                if self._pose:
                    self._pose = self._pose._replace(translation=self._pose.translation + self._pose.rotation.dot(t), rotation=R.dot(self._pose.rotation))
                else:
                    self._pose = Pose(R, t)
                self._last_pose = Pose(R, t)
            else:
                print 'not found'

            if self.debug:
                cv2.rectangle(self.features, (0, 0), (600, 600), (0, 0, 0), -1)

                scale = max(max(p[0], p[2]) for p in last_points_3d)
                yscale = max(p[1] for p in last_points_3d)
                scale = 50000
                yscale = 1000
                for p in last_points_3d:
                    c = max(0, 255 - int(200 * p[1] / yscale))
                    cv2.circle(self.features, (300 + int(300 * p[0] / scale), 600 - int(600 * p[2] / scale)), 1, (0, 0, c))

                for a, b in zip(last_points_3d, new_points_3d):
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

                MIN = min(a[2] - b[2] for a, b in zip(last_points_3d, new_points_3d))
                MAX = max(a[2] - b[2] for a, b in zip(last_points_3d, new_points_3d))

                text = "dZ min: %f" % MIN
                cv2.putText(self.features, text, (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "dZ max: %f" % MAX
                cv2.putText(self.features, text, (20, 85), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                text = "dZ pos: %f" % (dZ)
                cv2.putText(self.features, text, (20, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                text = "reproj error inliers: %f" % reproj_error_inliers
                cv2.putText(self.features, text, (300, 85), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "reproj error total  : %f" % reproj_error
                cv2.putText(self.features, text, (300, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                imga = self._last_frame.images[0].image
                imgb = frame.images[0].image
                imga = imga.get() if isinstance(imga, cv2.UMat) else imga
                imgb = imgb.get() if isinstance(imgb, cv2.UMat) else imgb
                img = np.concatenate((imga, imgb), axis=0)
                h = imga.shape[0]

                for i, P in enumerate(zip(last_points_2d, last_points_2d_right)):
                    l, r = P
                    cv2.line(img, (int(l[0]), int(l[1])), (int(r[0]), int(r[1])), (0, 255 if i in inliers else 0, 0 if i in inliers else 255))
                for i, P in enumerate(zip(new_points_2d, new_points_2d_right)):
                    l, r = P
                    cv2.line(img, (int(l[0]), int(l[1]) + h), (int(r[0]), int(r[1]) + h), (0, 255 if i in inliers else 0, 0 if i in inliers else 255))

                for l, n in zip(last_points_2d, new_points_2d):
                    cv2.line(img, (int(l[0]), int(l[1])), (int(n[0]), int(n[1]) + h), (255, 0, 0))

                for p in projected_2d:
                    cv2.circle(img, (int(p[0][0]), int(p[0][1]) + h), 3, (0, 0, 255))

                cv2.imshow("feature matches", img)
                cv2.imshow("3D points", self.features)

                if reproj_error_inliers > 5:
                    print 'fail'
                    print last_points_3d.tolist()
                    print new_points_2d.tolist()
                    if use_rt:
                        print _r.tolist()
                        print _t.tolist()
                    print self._camera.left.matrix.tolist()

                    cv2.waitKey(0)

        self._last_frame = frame
        self._last_3dfeatures = stereo_features
        return frame, self._pose

    def _calculate_3d(self, featuresA, featuresB):
        """ Finds stereo correspondances and triangulates the points.
        will return corresponding left/right points and triangulated points
        :return: (left, right, 3d points) or None
        """
        kpsA, descriptorsA = featuresA
        kpsB, descriptorsB = featuresB
        matches = self._match_features(descriptorsA, descriptorsB, self._feature_type, self._ratio, self._distance_thresh, self._min_matches)

        if matches is None or not matches:
            return None

        umat_descriptors = isinstance(descriptorsA, cv2.UMat)
        if umat_descriptors:
            descriptorsA = descriptorsA.get()
            descriptorsB = descriptorsB.get()

        kpsA = [kpsA[m.queryIdx] for m in matches]
        kpsB = [kpsB[m.trainIdx] for m in matches]
        mask = [abs(a.pt[1] - b.pt[1]) < self._dY and 0 < a.pt[0] - b.pt[0] < self._dX for a, b in zip(kpsA, kpsB)]

        dA = np.array([descriptorsA[m.queryIdx] for M, m in zip(mask, matches) if M], dtype=descriptorsA.dtype)
        dB = np.array([descriptorsB[m.trainIdx] for M, m in zip(mask, matches) if M], dtype=descriptorsB.dtype)

        kpsA = [p for M, p in zip(mask, kpsA) if M]
        kpsB = [p for M, p in zip(mask, kpsB) if M]

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
        mask = [abs(a[2] - b[2]) < dZ for a, b in zip(last_points_3d, new_points_3d)]

        if sum(mask) < self._min_matches:
            return None

        new_points_3d = np.float32([p for M, p in zip(mask, new_points_3d) if M])
        last_points_3d = np.float32([p for M, p in zip(mask, last_points_3d) if M])
        new_points_2d = np.float32([new_features[0][m.trainIdx] for M, m in zip(mask, matches) if M])
        new_points_2d_right = np.float32([new_features[1][m.trainIdx] for M, m in zip(mask, matches) if M])
        last_points_2d = np.float32([last_features[0][m.queryIdx] for M, m in zip(mask, matches) if M])
        last_points_2d_right = np.float32([last_features[1][m.queryIdx] for M, m in zip(mask, matches) if M])

        return last_points_2d, last_points_3d, new_points_2d, new_points_3d, last_points_2d_right, new_points_2d_right

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