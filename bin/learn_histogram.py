# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from argparse import ArgumentParser
from EasyVision.processors import HistogramBackprojection
from EasyVision.vision import VideoCapture


class App(object):
    def __init__(self, video_src):
        self.cam = cv2.VideoCapture(video_src)
        _ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')
        cv2.setMouseCallback('camshift', self.onmouse)

        self.selection = None
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None

    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            if xmax - xmin > 10 and ymax - ymin > 10:
                self.selection = (xmin, ymin, xmax, ymax)
        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 2
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    def run(self):
        while True:
            _ret, self.frame = self.cam.read()
            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            if self.selection:
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv[y0:y1, x0:x1]
                hist = HistogramBackprojection.calculate_histogram(self.frame[y0:y1, x0:x1])
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
                prob = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]
                try:
                    cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                except:
                    print(track_box)

            cv2.imshow('camshift', vis)

            ch = cv2.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser(description="Color histogram learning tool")
    parser.add_argument("camera", type=int, help="Camera device ID/folder")
    parser.add_argument("-f", "--file", default="histogram.json", help="Output filename of the learned histogram")
    parser.add_argument("-i", "--size", default="640,480", help="Frame width and height")
    parser.add_argument("-t", "--test", const=True, default=False, action='store_const', help="Test histogram")

    args = parser.parse_args()
    size = tuple(int(i) for i in args.size.split(','))
    try:
        camera = int(args.camera)
    except:
        camera = args.camera
    app = App(args.camera)
    app.run()

    with open(args.file, "w") as f:
        json.dump(app.hist.tolist(), f)
