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
from PIL import Image, ImageDraw, ImageFont

class ONNXRuntimeImageClassification:
    """Image classification class for ONNX Runtime
    """
    def __init__(self, config_filename):

        f = open(config_filename)
        data = json.load(f)

        self.model_filename = str(data["MODEL_FILENAME"])
        self.label_filename = str(data["LABELS_FILENAME"])
        self.video_inp = str(data["Input"])
        self.model_inp_width = int(data["ScaleWidth"])
        self.model_inp_height = int(data["ScaleHeight"])
        self.disp = int(data["display"])
        self.input_format = str(data["InputFormat"])
        self.mean_vec = np.array(data["mean_vec"]) 
        self.stddev_vec = np.array(data["stddev_vec"]) 
        self.labels = []

        with open(self.label_filename, 'r') as f:
            self.labels = [l.strip() for l in f.readlines()]

        self.session = onnxruntime.InferenceSession(self.model_filename, None)
        self.input_name = self.session.get_inputs()[0].name

    def load_labels(self, path):
        with open(path) as f:
            for cnt, line in enumerate(f):
                self.labels.append(line.rstrip("\n"))
    
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
        raw_result = self.session.run([], {input_name: input_data})[1]
        for i in raw_result:
            label_dict = i
    
        predictions = []
        for key in self.labels:
            predictions.append(label_dict[key])
        return predictions

    def softmax(self, x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def postprocess(self, result):
        return self.softmax(np.array(result)).tolist()

def main(config_filename):

    ic_model = ONNXRuntimeImageClassification(config_filename)

    if ic_model.video_inp == 'cam':
        cap = cv2.VideoCapture(0)
        # Reading widht and height details
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if cap == None:
           print("Error: Input Camera device not found/detected")
    else:
        print("Error: Invalid input argument/source")
        sys.exit(0)

    #ic_model.load_labels(ic_model.label_filename)
    color = (255, 255 , 255)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    while(True):
       # Caputre frame-by-frame
       ret, frame = cap.read()
       re_size = cv2.resize(frame, (ic_model.model_inp_width, ic_model.model_inp_height))

       predictions = ic_model.predict_image(re_size)

       res = ic_model.postprocess(predictions)
       idx = np.argmax(res)

       frame = cv2.putText(frame, ic_model.labels[idx], (15, 15), 
                           font, fontScale, color, thickness, cv2.LINE_AA)

       if ic_model.disp == 1:
          cv2.imshow("Inference results", frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    # when everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} config_filename'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
