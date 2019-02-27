from EasyVision.engine.topologicalmap import TopologicalMap
from tests.common import *


def main():
    traj = np.zeros((600, 600, 3), dtype=np.uint8)
    pose = '00'
    feature_type = 'ORB'

    images_kitti = ['d:/datasets/data_odometry_gray/dataset/sequences/{}/image_0/{}.png'.format(pose, str(i).zfill(6)) for i in range(NUM_IMAGES)]
    gt_path_kitti = "d:/datasets/data_odometry_gray/dataset/poses/{}.txt".format(pose)

    with open(gt_path_kitti) as f:
        ground_truth = [[float(i) for i in line.split()] for line in f.readlines()]

    error = 0
    cam = CalibratedCamera(ImageTransform(ImagesReader(images_kitti), ocl=True), camera_kitti, display_results=True)
    occmap = TopologicalMap((1200, 1200), "D:/Projects/python/EasyVision/EasyVision/engine/orbvoc.dbow3", scale=2, min_y=-.1, max_d=50, display_results=True)
    pose = Pose(0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[290], [0], [90]])
    with VisualOdometry2DEngine(cam, _map=occmap, pose=pose, display_results=True, feature_type=feature_type) as engine:
        for img_id, _ in enumerate(images_kitti):
            true_x = ground_truth[img_id][3]
            true_y = ground_truth[img_id][7]
            true_z = ground_truth[img_id][11]

            if img_id > 0:
                scale = np.sqrt((true_x - x_prev) ** 2 + (true_y - y_prev) ** 2 + (true_z - z_prev) ** 2)
                x_prev, y_prev, z_prev = true_x, true_y, true_z
            else:
                scale = 1.0
                x_prev, y_prev, z_prev = true_x, true_y, true_z

            frame, pose = engine.compute(absolute_scale=scale)
            if pose:
                t = pose.translation

                error += np.sqrt((true_x - t[0] + 290) ** 2 + 0 * (true_y - t[1]) ** 2 + (true_z - t[2] + 90) ** 2)

                draw_x, draw_y = int(t[0]), int(t[2])
                dtrue_x, dtrue_y = int(true_x) + 290, int(true_z) + 90

                cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0))
                cv2.circle(traj, (dtrue_x, dtrue_y), 1, (0, 0, 255))
                cv2.rectangle(traj, (0, 0), (600, 60), (0, 0, 0), -1)
                text = "pose: x=%2fm y=%2fm z=%2fm" % (t[0], t[1], t[2])
                cv2.putText(traj, text, (20, 11), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "true: x=%2fm y=%2fm z=%2fm" % (true_x, true_y, true_z)
                cv2.putText(traj, text, (20, 22), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "scale: %2f" % scale
                cv2.putText(traj, text, (20, 33), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "cumulative error: %2f " % error
                cv2.putText(traj, text, (20, 44), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

            cv2.imshow('Trajectory', traj)
            if cv2.waitKey(1) == 27:
                break

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
