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

 Author: Pradeep, Sakhamoori <pradeep.sakhamoori@intel.com>
 
"""

import sys, os
import onnxruntime
import numpy as np
import cv2
import json
import iot_hub_manager
import time
import datetime

from VideoStream import VideoStream
from object_detection import ObjectDetection
from iot_hub_manager import IotHubManager
from iothub_client import IoTHubTransportProvider, IoTHubError
from onnxruntime.capi.onnxruntime_pybind11_state import RunOptions

class ONNXRuntimeModelDeploy(ObjectDetection):
    """Object Detection class for ONNX Runtime
    """
    def __init__(self, manifest):
        # Default system params
        self.video_inp = "cam"
        self.render = 1

        self.m_parser(manifest)

    def m_parser(self, manifest):

        m_file = open(manifest)
        data = json.load(m_file)

         # cvexport manifest prameters
        self.domain_type = str(data["DomainType"])

        # default anchors
        if str(self.domain_type) == "ObjectDetection":
           # Model dependent params for ObjectDetection (default)
           self.model_inp_width = 416
           self.model_inp_height = 416
           self.input_format = "RGB"
           self.iou_threshold = 0.45
           self.conf_threshold = 0.5
           self.anchors = np.array([[1.08, 1.19], [3.42, 4.41],  [6.63, 11.38],  [9.42, 5.11],  [16.62, 10.52]])
           if "IouThreshold" in data:
              self.iou_threshold = data["IouThreshold"]
           if "ConfThreshold" in data:
              self.conf_threshold = data["ConfThreshold"]

        elif str(self.domain_type) == "ImageClassification":
           # Model dependent params for ImageClassification (default)
           self.mean_vec = [0.485, 0.456, 0.406],
           self.stddev_vec = [0.229, 0.224, 0.225],
           self.model_inp_width = 224
           self.model_inp_height = 24
           if "MeanVec" in data:
              self.mean_vec = data["MeanVec"]
           if "StddevVec" in data:
              self.stddev_vec = data["StddevVec"]
        else:
           print("Error: No matching DaominType: should be ObjectDetection/ImageClassificaiton \n")
           print("Exiting.....!!!! \n")
           sys.exit(0)

        self.platform = str(data["Platform"])
        self.model_filename = str(data["ModelFileName"])
        self.label_filename = str(data["LabelFileName"])

        if "InputStream" in data:
            self.video_inp = str(data["InputStream"])
        if "ScaleWidth" in data:
            self.model_inp_width = int(data["ScaleWidth"])
        if "ScaleHeight" in data:
            self.model_inp_height = int(data["ScaleHeight"])
        if "RenderFlag" in data:
            self.render = int(data["RenderFlag"])
        if "Anchors" in data:
            self.objdet_anchors = np.array(data["Anchors"])
        if "InputFormat" in data:
            self.input_format = str(data["InputFormat"])

        # Application parameters
        self.img_width = 0
        self.img_height = 0
        self.cap_handle = None
        self.vs = None
        self.session = None

        with open(str("./model/" + self.label_filename), 'r') as f:
            labels = [l.strip() for l in f.readlines()]

        super(ONNXRuntimeModelDeploy, self).__init__(labels)
        print("\n Triggering Inference...")

        self.session = onnxruntime.InferenceSession(str("./model/" + self.model_filename))

        print("\n Started Inference...")
        self.input_name = self.session.get_inputs()[0].name
        if self.render == 0:
           print("Press Ctl+C to exit...")
  
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))
        start = time.time()
        outputs = self.session.run(None, {self.input_name: inputs})
        end = time.time()
        inference_time = end - start
        return np.squeeze(outputs).transpose((1,2,0)), inference_time

    def create_video_handle(self):

        web_cam_found = False
        for i in range(4):
            if os.path.exists("/dev/video"+str(i)):
              web_cam_found = True
              break

        if web_cam_found:
           usb_video_path = "/dev/video"+str(i)
        else:
           print("\n Error: Input Camera device not found/detected")
           print("\n Exisiting inference...")
           sys.exit(0)

        self.vs = VideoStream(usb_video_path).start()

        # Reading widht and height details
        self.img_width = int(self.vs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(self.vs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def model_inference(self, iot_hub_manager):

        self.create_video_handle()

        while self.vs.stream.isOpened():

            if iot_hub_manager.setRestartCamera == True:
               #self.cap_handle.release()), 1)
               #RunOptions.terminate = True
               self.vs.stream.release()
               cv2.destroyAllWindows()

               if os.path.exists('./model/cvexport.manifest'):
                   print("\n Reading cvexport.config file from model folder")
                   config_filename = "./model/cvexport.manifest"
                   self.__init__(config_filename)
                   self.create_video_handle()
               elif os.path.exists("cvexport.manifest"):
                   config_filename = "cvexport.manifest"
                   print("\n Reading cvexport.manifest file from default base folder")
                   self.__init__(config_filename)
                   self.create_video_handle()
               else:
                   print("\n ERROR: cvexport.manifest not found check root/model dir")
                   print("\n Exiting inference....")
                   sys.exit(0)
               iot_hub_manager.setRestartCamera = False

            # Caputre frame-by-frame
            frame = self.vs.read()
            if str(self.domain_type) == "ObjectDetection":
               print("Calling predict_image \n")
               predictions, infer_time = self.predict_image(frame)

            for d in predictions:
                x = int(d['boundingBox']['left'] * self.img_width)
                y = int(d['boundingBox']['top'] * self.img_height)
                w = int(d['boundingBox']['width'] * self.img_width)
                h = int(d['boundingBox']['height'] * self.img_height)

                x_end = x+w
                y_end = y+h

                start = (x,y)
                end = (x_end,y_end)

                if 0.45 < d['probability']:
                    frame = cv2.rectangle(frame,start,end, (0, 255, 255), 2)

                    out_label = str(d['tagName'])
                    score = str(int(d['probability']*100))
                    cv2.putText(frame, out_label, (x-5, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, score, (x+w-50, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                    message = { "Label": out_label,
                                "Confidence": score,
                                "Position": [x, y, x_end, y_end],
                                "TimeStamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Send message to IoT Hub
                    if iot_hub_manager is not None:
                       iot_hub_manager.send_message_to_upstream(json.dumps(message))

            cv2.putText(frame, 'FPS: {}'.format(1.0/infer_time), (10,40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)

            if self.render == 1:
                # Displaying the image
                cv2.imshow("Inference results", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   break

        # when everything done, release the capture
        self.vs.__exit__(None, None, None)
        cv2.destroyAllWindows()

def main():

    manifest_file_path = "./model/cvexport.manifest"
    od_handle = ONNXRuntimeModelDeploy(manifest_file_path)

    # Adding iot support
    # Choose HTTP, AMQP or MQTT as transport protocol.  Currently only MQTT is supported.
    IOT_HUB_PROTOCOL = IoTHubTransportProvider.MQTT
    iot_hub_manager = IotHubManager(IOT_HUB_PROTOCOL)

    #print(" Starting model inference... ")
    od_handle.model_inference(iot_hub_manager)

if __name__ == '__main__':
    main()
