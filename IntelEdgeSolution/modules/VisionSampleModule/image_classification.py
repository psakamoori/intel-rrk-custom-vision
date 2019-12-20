"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
import sys
from onnx import numpy_helper
import urllib.request
import json
import cv2
import time

# display images in notebook
#from PIL import Image, ImageDraw, ImageFont

class ImageClassification(object):
    """Image classification class for ONNX Runtime
    """
    def __init__(self, data):

        # TBD Need to add error check
        self.platform = str(data["Platform"])
        self.model_filename = str(data["ModelFileName"])
        self.label_filename = str(data["LabelFileName"])

        if "InputStream" in data:
            self.video_inp = str(data["InputStream"])
        if "ScaleWidth" in data:
            self.model_inp_width = int(data["ScaleWidth"])
        else:
            self.model_inp_width = 224

        if "ScaleHeight" in data:
            self.model_inp_height = int(data["ScaleHeight"])
        else:
            self.model_inp_height = 24
        if "RenderFlag" in data:
            self.render = int(data["RenderFlag"])
        else:
            self.render = 1

        if "MeanVec" in data:
           self.mean_vec = data["MeanVec"]
        else:
           self.mean_vec = [0.485, 0.456, 0.406]

        if "StddevVec" in data:
           self.stddev_vec = data["StddevVec"]
        else:
           self.stddev_vec = [0.229, 0.224, 0.225]
           
        if "InputFormat" in data:
            self.input_format = str(data["InputFormat"])
        
        self.session = None
        self.onnxruntime_session_init()

    def onnxruntime_session_init(self):
        if self.session is not None:
           self.session = None

        #onnxruntime.capi.onnxruntime_pybind11_state.RunOptions = False

        with open(self.label_filename, 'r') as f:
            self.labels = [l.strip() for l in f.readlines()]

        self.session = onnxruntime.InferenceSession(self.model_filename, None)
        self.input_name = self.session.get_inputs()[0].name

    def load_labels(self, path):
        with open(path) as f:
            for cnt, line in enumerate(f):
                self.labels.append(line.rstrip("\n"))
        print("total_classes =", cnt)


    def preprocess(self, input_data):
        # convert the input data into the float32 input
        img_data = input_data.astype('float32')
        img_data = img_data.reshape(1, 3, self.model_inp_width, self.model_inp_height)

        #normalize
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
            norm_img_data[i,:,:] = (img_data[i,:,:]/255 - self.mean_vec[i]) / self.stddev_vec[i]
        return norm_img_data

    def predict_image(self, frame):
        image_data = np.array(frame).transpose(2, 0, 1)
        input_data = self.preprocess(image_data)
        input_name = self.session.get_inputs()[0].name 

        raw_result = {}
        start = time.time()
        raw_result = self.session.run([], {input_name: input_data})[1]
        end = time.time()
        inference_time = end - start
        for i in raw_result:
            label_dict = i
    
        predictions = []
        for key in self.labels:
            predictions.append(label_dict[key])
        return predictions, inference_time

    def softmax(self, x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def postprocess(self, result):
        return self.softmax(np.array(result)).tolist()

def main(config_filename):

    ic_model = ONNXRuntimeImageClassification(config_filename)
