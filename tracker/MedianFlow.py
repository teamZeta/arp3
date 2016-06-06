from numpy import *
import vot
from mftracker import *


class MedianFlow(object):
    def __init__(self, image, region):

        self.region = region
        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)
        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.img = image
        self.oldg = cv2.cvtColor(self.img, cv2.cv.CV_BGR2GRAY)
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)
        self.bb = [left, top, left + region.width, top + region.height]

    def set_position(self, position):
        self.position = (position[0], position[1])
        self.bb = [position[0]- self.size[0]/2, position[1]- self.size[1]/2, position[0]+ self.size[0]/2, position[1]+ self.size[1]/2]

    def set_region(self, region):
        self.position = (int(region.x + region.width / 2), int(region.y + region.height / 2))
        self.size = (region.width, region.height)
        self.bb = [self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2, self.position[0]+ self.size[0]/2, self.position[1]+ self.size[1]/2]

    def track(self, image):


        newg = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)

        # self.bb = [left, top, self.region.width, self.region.height]

        print(self.bb)
        if self.bb[0] < 0 or self.bb[1] < 0 or self.bb[1] >= image.shape[1] or self.bb[3] >= image.shape[0]:
            newbb, shift = fbtrack(self.oldg, newg, self.bb, 12, 12, 3, 12)
            self.bb = newbb
        self.oldg = newg
        self.position = (self.bb[0] + self.size[0] / 2, self.bb[1] + self.size[1] / 2)
        self.window = max(self.bb[2] - self.bb[0], self.bb[3] - self.bb[1]) * 2

        left = int(max(round(self.position[0] - float(self.window) / 2), 0))
        top = int(max(round(self.position[1] - float(self.window) / 2), 0))
        right = int(min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1))
        bottom = int(min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1))

        if self.bb[0] < 0 or self.bb[1] < 0 or self.bb[1] >= image.shape[1]-1 or self.bb[3] >= image.shape[0]-1:
            print("NOTER JE PRISLO")
            return vot.Rectangle(1, 1, self.size[0], self.size[1])

        return vot.Rectangle(self.bb[0], self.bb[1], self.bb[2]-self.bb[0], self.bb[3]-self.bb[1])
