import cv2
import numpy as np
from numpy import *
import vot
import random
from mondrianforest_utils import reset_random_seed, precompute_minimal
from mondrianforest import MondrianForest


class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


class ORF(object):
    def __init__(self, img, region):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        image = image.astype(int) / 4
        h, s, v = cv2.split(image)
        s += 45
        v += 109
        image = h  # cv2.merge((h, s, v))
        self.window = max(region.width, region.height) * 2

        left = int(max(region.x, 0))
        top = int(max(region.y, 0))

        right = int(min(region.x + region.width, image.shape[1] - 1))
        bottom = int(min(region.y + region.height, image.shape[0] - 1))

        if (right - left) % 2 != 0:
            right -= 1
        if (bottom - top) % 2 != 0:
            bottom -= 1

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)
        self.old_img = image
        self.pos = np.array([image[int(top):int(bottom), int(left):int(right)].copy().tolist()])
        self.neg = np.array([])
        self.st_neg = 0

        vred = [(0, 0)]
        infloop = 0
        while (1):
            l = random.randint(-int((right - left) * 0.03), int((right - left) * 0.03))
            t = random.randint(-int((bottom - top) * 0.03), int((bottom - top) * 0.03))
            if l + left >= 0 and l + right < image.shape[1] and t + top >= 0 and t + bottom < image.shape[0]:
                vred += [(l, t)]
            if len(vred) > 15 or infloop > 10000:
                break
            infloop += 1

        self.pos = np.array(
            [np.array(self.old_img[top + int(t2):bottom + int(t2), left + int(l2):right + int(l2)].copy().tolist()) for
             (l2, t2) in vred])
        vred = []
        infloop = 0
        while (1):
            l = random.randint(int((right - left) / 2), int(image.shape[1] - (right - left) / 2 - 1))
            t = random.randint(int((bottom - top) / 2), int(image.shape[0] - (bottom - top) / 2 - 1))
            if abs(l - left - (right - left) / 2) > (right - left) or abs(t - top - (bottom - top) / 2) > (
                        bottom - top):
                vred += [(l, t)]
            if len(vred) > 45 or infloop > 10000:
                break
            infloop += 1
        self.neg = np.array(
            [np.array(self.old_img[int(t2) - (bottom - top) / 2:int(t2) + (bottom - top) / 2,
                      int(l2) - (right - left) / 2:int(l2) + (right - left) / 2].copy().tolist()) for (l2, t2) in vred])
        print("pred update")
        print("neg" + str(len(self.neg)))
        print("pos" + str(len(self.pos)))

        set = {'optype': 'class', 'verbose': 1, 'draw_mondrian': 0, 'perf_dataset_keys': ['train', 'test'],
               'data_path': '../../process_data/', 'dataset': 'toy-mf', 'tag': '', 'alpha': 0, 'bagging': 0,
               'select_features': 0, 'smooth_hierarchically': 1, 'normalize_features': 1, 'min_samples_split': 2,
               'save': 0, 'discount_factor': 10, 'op_dir': 'results', 'init_id': 1, 'store_every': 0,
               'perf_store_keys': ['pred_prob'], 'perf_metrics_keys': ['log_prob', 'acc'], 'budget': -1.0,
               'n_mondrians': 10, 'debug': 0, 'n_minibatches': 1, 'name_metric': 'acc', 'budget_to_use': inf}
        self.settings = Map(set)
        reset_random_seed(self.settings)

        x_trainp = np.array([np.bincount(x.flatten().astype(int), minlength=45) for x in self.pos])
        x_trainn = np.array([np.bincount(x.flatten().astype(int), minlength=45) for x in self.neg])
        if len(self.neg) > 0:
            x_train = np.append(x_trainp, x_trainn, axis=0)
            self.st_neg = 1
        else:
            x_train = x_trainp
            self.st_neg = 0


        self.data = {'n_dim': 1, 'x_test':
            array([x_train[5]]),
                     'x_train': array(x_train),
                     'y_train': array(
                         np.ones(len(self.pos)).astype(int).tolist() + np.zeros(len(self.neg)).astype(int).tolist()),
                     'is_sparse': False, 'n_train': len(x_train), 'n_class': 2,
                     'y_test': array([]),
                     'n_test': 0}

        self.param, self.cache = precompute_minimal(self.data, self.settings)
        self.mf = MondrianForest(self.settings, self.data)
        self.mf.fit(self.data, array(range(0, len(x_train))), self.settings, self.param, self.cache)

    def set_region(self, position):
        self.position = position

    def reset_position(self, pos):
        self.position = pos

    def set_position(self, position):
        self.position = (position[0], position[1])
        self.size = [position[0] - self.size[0] / 2, position[1] - self.size[1] / 2, position[0] + self.size[0] / 2,
                     position[1] + self.size[1] / 2]

    def set_region(self, region):
        self.position = (int(region.x + region.width / 2), int(region.y + region.height / 2))
        self.size = (region.width, region.height)

    def updateTree(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        image = image.astype(int) / 4
        h, s, v = cv2.split(image)
        s += 45
        v += 109
        image = h  # cv2.merge((h, s, v))

        t = len(self.template) / 2
        l = len(self.template[0]) / 2
        left = self.position[0] - l
        top = self.position[1] - t
        right = self.position[0] + l
        bottom = self.position[1] + t

        vred = [(0, 0)]
        infloop = 0
        while (1):
            l2 = random.randint(-int((right - left) * 0.03), int((right - left) * 0.03))
            t2 = random.randint(-int((bottom - top) * 0.03), int((bottom - top) * 0.03))
            if l2 + left >= 0 and l2 + right < image.shape[1] and t2 + top >= 0 and t + bottom < image.shape[0]:
                vred += [(l2, t2)]
            if len(vred) > 5 or infloop > 10000:
                break
            infloop += 1
        self.pos = np.array(
            [np.array(image[top + t2:bottom + t2, left + l2:right + l2].copy().tolist()) for (l2, t2) in vred])
        vred = []
        infloop = 0
        while (1):
            l2 = random.randint((right - left) / 2, image.shape[1] - (right - left) / 2 - 1)
            t2 = random.randint((bottom - top) / 2, image.shape[0] - (bottom - top) / 2 - 1)
            if abs(l2 - left - (right - left) / 2) > (right - left) and abs(t2 - top - (bottom - top) / 2) > (
                        bottom - top):
                vred += [(l2, t2)]
            if len(vred) > 15 or infloop > 10000:
                break
            infloop += 1

        self.neg = np.array(
            [np.array(image[t2 - (bottom - top) / 2:t2 + (bottom - top) / 2,
                      l2 - (right - left) / 2:l2 + (right - left) / 2, ].tolist()) for (l2, t2) in vred])

        x_trainp = np.array([np.bincount(x.flatten().astype(int), minlength=45) for x in self.pos])
        x_trainn = np.array([np.bincount(x.flatten().astype(int), minlength=45) for x in self.neg])

        if len(self.neg) > 0:
            x_train = np.append(x_trainp, x_trainn, axis=0)
        else:
            x_train = x_trainp

        self.data['x_train'] = np.append(self.data['x_train'], array(x_train), axis=0)
        self.data['y_train'] = np.append(self.data['y_train'], array(
            np.ones(len(self.pos)).astype(int).tolist() + np.zeros(len(self.neg)).astype(int).tolist()))
        self.mf.partial_fit(self.data,
                            array(range(len(self.data['x_train']) - len(x_train), len(self.data['x_train']))),
                            self.settings, self.param, self.cache)

    def track(self, img):
        if self.st_neg == 0:
            return [0, vot.Rectangle(1, 1, self.size[0], self.size[1])]
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        image = image.astype(int) / 4
        h, s, v = cv2.split(image)
        s += 45
        v += 109
        image2 = h  # cv2.merge((h, s, v))
        left = int(max(round(self.position[0] - self.size[0]), 0))
        top = int(max(round(self.position[1] - self.size[1]), 0))
        right = int(min(round(self.position[0] + self.size[0]), image2.shape[1] - 1))
        bottom = int(min(round(self.position[1] + self.size[1]), image2.shape[0] - 1))
        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [0, vot.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2,
                                     self.size[0],
                                     self.size[1])]

        cut = image2[top:bottom, left:right]

        t = int(self.size[1] / 2)
        l = int(self.size[0] / 2)
        weights_prediction = np.ones(self.settings.n_mondrians) * 1.0 / self.settings.n_mondrians
        predicta = []
        for i in range(t, int(bottom - top - t), 5):
            for j in range(l, right - left - l, 5):
                imclass = cut[i - t:i + t, j - l:j + l]
                pred = self.mf.evaluate_predictions(self.data,
                                                    array([np.bincount(imclass.flatten().astype(int), minlength=45)]),
                                                    [1],
                                                    self.settings, self.param, weights_prediction, False)[0]
                predicta += [(pred['pred_prob'][0][1], i, j)]

        if len(predicta) < 1:
            return [0, vot.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2,
                                     self.size[0],
                                     self.size[1])]

        terk = predicta[0]
        for i in range(0, len(predicta)):
            if terk[0] < predicta[i][0]:
                terk = predicta[i]

        t = int(self.size[1] / 2 * 0.9)
        l = int(self.size[0] / 2 * 0.9)
        if int(top + terk[1] - t) >= 0 and int(top + terk[1] + t) < image2.shape[0] and int(
                                left + terk[2] - l) >= 0 and int(left + terk[2] + l) < image2.shape[0]:
            imclass = image2[int(top + terk[1] - t):int(top + terk[1] + t),
                      int(left + terk[2] - l):int(left + terk[2] + l)]
            predmin = \
                self.mf.evaluate_predictions(self.data,
                                             array([np.bincount(imclass.flatten().astype(int), minlength=45)]), [1],
                                             self.settings, self.param, weights_prediction, False)[0]
        else:
            predmin = None

        t = int(self.size[1] / 2 * 1.1)
        l = int(self.size[0] / 2 * 1.1)
        if int(top + terk[1] - t) >= 0 and int(top + terk[1] + t) < image2.shape[0] and int(
                                left + terk[2] - l) >= 0 and int(left + terk[2] + l) < image2.shape[0]:
            imclass = image2[int(top + terk[1] - t):int(top + terk[1] + t),
                      int(left + terk[2] - l):int(left + terk[2] + l)]
            predmax = \
                self.mf.evaluate_predictions(self.data,
                                             array([np.bincount(imclass.flatten().astype(int), minlength=45)]),
                                             [1],
                                             self.settings, self.param, weights_prediction, False)[0]
        else:
            predmax = None

        if predmax != None and predmin != None and predmax['pred_prob'][0][1] > predmin['pred_prob'][0][1]:
            if predmax['pred_prob'][0][1] > terk[0]:
                self.size = (int(self.size[0] * 1.1), int(self.size[1] * 1.1))

        elif predmin != None:
            if predmin['pred_prob'][0][1] > terk[0]:
                self.size = (int(self.size[0] * 0.9), int(self.size[1] * 0.9))
        self.position = (left + terk[2], top + terk[1])

        return [terk[0], vot.Rectangle(left + terk[2] - l, top + terk[1] - t, self.size[0], self.size[1])]
