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
from image_classification import ImageClassification
from iot_hub_manager import IotHubManager
from iothub_client import IoTHubTransportProvider, IoTHubError
from onnxruntime.capi.onnxruntime_pybind11_state import RunOptions

# Adding iot support
# Choose HTTP, AMQP or MQTT as transport protocol.  Currently only MQTT is supported.
IOT_HUB_PROTOCOL = IoTHubTransportProvider.MQTT
iot_hub_manager = IotHubManager(IOT_HUB_PROTOCOL)

class ONNXRuntimeModelDeploy(ObjectDetection, ImageClassification):
    """Object Detection class for ONNX Runtime
    """
    def __init__(self, manifest):
        # Default system params
        self.video_inp = "cam"
        self.render = True

        # Application parameters
        self.img_width = 0
        self.img_height = 0
        self.cap_handle = None
        self.vs = None
        self.session = None

        self.m_parser(manifest)

    def m_parser(self, manifest):

        m_file = open(manifest)
        data = json.load(m_file)

         # cvexport manifest prameters
        self.domain_type = str(data["DomainType"])
        print("Domain Type:", self.domain_type)

        # default anchors
        if str(self.domain_type) == "ObjectDetection":
           objdet = ObjectDetection(data, None)
           self.model_inference(objdet, iot_hub_manager, 1)
        elif str(self.domain_type) == "Classification":
           imgcls = ImageClassification(data)
           self.model_inference(imgcls, iot_hub_manager, 0)
        else:
           print("Error: No matching DaominType: should be ObjectDetection/Classificaiton \n")
           print("Exiting.....!!!! \n")
           sys.exit(0)

    #def predict(self, preprocessed_image):
    #    inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
    #    inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))
    #    start = time.time()
    #    outputs = self.session.run(None, {self.input_name: inputs})
    #    end = time.time()
    #    inference_time = end - start
    #    return np.squeeze(outputs).transpose((1,2,0)), inference_time

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

    def model_inference(self, obj, iot_hub_manager, pp_flag):

        self.create_video_handle()
        while self.vs.stream.isOpened():

            if iot_hub_manager.setRestartCamera == True:
               iot_hub_manager.setRestartCamera = False
               #self.cap_handle.release()
               obj.session = None
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
               #iot_hub_manager.setRestartCamera = False

            # Caputre frame-by-frame
            frame = self.vs.read()
            predictions, infer_time = obj.predict_image(frame)

            # if Object Detection
            if pp_flag:
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
            else:  #Postprocessing for Classificaton model

                res = obj.postprocess(predictions)
                idx = np.argmax(res)

                frame = cv2.putText(frame, obj.labels[idx], (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                message = { "Label": obj.labels[idx],
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
    ONNXRuntimeModelDeploy(manifest_file_path)

if __name__ == '__main__':
    main()
