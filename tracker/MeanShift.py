import vot
import cv2
import numpy as np



class MeanShift(object):
    def __init__(self, image, region):
        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)
        self.window = max(region.width, region.height) * 2

        roi = image[int(top):int(bottom), int(left):int(right), :]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((180., 255., 255.)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.bb = [left, top, region.width, region.height]

    def set_position(self, position):
        self.position = (position[0], position[1])
        self.bb = [position[0] - self.size[0] / 2, position[1] - self.size[1] / 2, self.size[0],
                   self.size[1]]

    def set_region(self, region):
        self.position = (int(region.x + region.width / 2), int(region.y + region.height / 2))
        self.size = (region.width, region.height)
        self.bb = [int(self.position[0] - self.size[0] / 2), int(self.position[1] - self.size[1] / 2),
                   int(self.size[0]),
                   int(self.size[1])]

    def track(self, image):
        left = int(max(round(self.position[0] - float(self.window) / 2), 0))
        top = int(max(round(self.position[1] - float(self.window) / 2), 0))

        right = int(min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1))
        bottom = int(min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1))

        if right - left < self.size[1] or bottom - top < self.size[0]:
            return vot.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0],
                                 self.size[1])

        img = image[top:bottom, left:right]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, (
        int(self.bb[0] - left), int(self.bb[1] - top), int(self.size[0]), int(self.size[1])), self.term_crit)
        self.position = (
        left + track_window[0] + int(track_window[2] / 2), top + track_window[1] + int(track_window[3] / 2))
        self.bb = [left + track_window[0], top + track_window[1], track_window[2], track_window[3]]
        self.size = (track_window[2], track_window[3])
        return vot.Rectangle(left + track_window[0], top + track_window[1], track_window[2], track_window[3])
