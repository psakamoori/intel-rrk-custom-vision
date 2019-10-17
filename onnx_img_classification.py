"""

 Copyright (c) 2018 Intel Corporation



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
from onnx import numpy_helper
import urllib.request
import json
import time


model_path = 'models/image_classification/dog_n_cat/model.onnx'
labels_path = 'models/image_classification/dog_n_cat/labels.txt'
image_path = 'samples/dog1.jpg'

# display images in notebook
from PIL import Image, ImageDraw, ImageFont

# Run the model on the backend
session = onnxruntime.InferenceSession(model_path, None)

# get the name of the first input of the model
input_name = session.get_inputs()[0].name  
input_shape = session.get_inputs()[0].shape
output_shape = session.get_outputs()[0].shape  

print('Input Name:', input_name)
print('Input Shape:', input_shape)
print('Output Shape:', output_shape)

def load_labels(path):
    with open(path) as f:
        #data = json.load(f)
        #line = f.readline()
        data = []
        #while line:
        #    data.append(line)
        #    line = f.readline()
        for cnt, line in enumerate(f):
            data.append(line.rstrip("\n"))
    return np.asarray(data)
    #return data

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')
    img_data = img_data.reshape(1, 3, 224, 224)

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

labels = load_labels(labels_path)
image = Image.open(image_path)

print("Image size: ", image.size)
image_data = np.array(image).transpose(2, 0, 1)
input_data = preprocess(image_data)

start = time.time()
raw_result = {}
raw_result = session.run([], {input_name: input_data})[1]
end = time.time()

print("raw_result", raw_result)
for i in raw_result:
  label_dict = i

output = []
for key in labels:
    output.append(label_dict[key])

res = postprocess(output)

inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(res)

print('========================================')
print('Final top prediction is: ' + labels[idx])
print('========================================')

print('========================================')
print('Inference time: ' + str(inference_time) + " ms")
print('========================================')

sort_idx = np.flip(np.squeeze(np.argsort(res)))
print('============ Top 5 labels are: ============================')
print(labels[sort_idx[:5]])
print('===========================================================')
