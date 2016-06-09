# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import os.path
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            '/home/boka/arp/vot-toolkit/tracker/examples/python/tracker', 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
    # if not tf.gfile.Exists(image):
    #    tf.logging.fatal('File does not exist %s', image)
    # image_data = tf.gfile.FastGFile(image, 'rb').read()
    create_graph()

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg:0': image[:, :, 0:3]})
        predictions = np.squeeze(predictions)
        return predictions


def get_closest(conn, images, root=None):
    prvi = 0
    min = 0
    id_min = 3
    stevec = -1
    for image in images:
        if len(image) > 0:
            with tf.Graph().as_default():
                if prvi == 0:
                    root = run_inference_on_image(image)
                    dist = [[0]]
                    prvi = 1
                else:
                    prvi += 1
                    predictions = run_inference_on_image(image)
                    dist = cosine_similarity(predictions, root)
                    print(dist)

                if prvi == 2 or min > dist[0][0]:
                    prvi = 2
                    min = dist
                    id_min = stevec
                stevec += 1
    print("id = " + str(id_min))
    conn.send([id_min])
    conn.close()
