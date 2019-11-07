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

 Usage : $python3 main.py
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

from object_detection import ObjectDetection
from iot_hub_manager import IotHubManager
from iothub_client import IoTHubTransportProvider, IoTHubError

# Choose HTTP, AMQP or MQTT as transport protocol.  Currently only MQTT is supported.
IOT_HUB_PROTOCOL = IoTHubTransportProvider.MQTT

# Disable sending D2C messages to IoT Hub to prevent consuming network bandwidth
iot_hub_manager = None


class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime
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
        self.anchors = np.array(data["Anchors"])
        self.iou_threshold = data["IOU_THRESHOLD"]
        self.input_format = str(data["InputFormat"])

        with open(self.label_filename, 'r') as f:
            labels = [l.strip() for l in f.readlines()]

        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        print("\n Triggering Inference...")
        self.session = onnxruntime.InferenceSession(self.model_filename)
        print("\n Started Inference...")
        self.input_name = self.session.get_inputs()[0].name 
        if self.disp == 0:
           print("Press Ctl+C to exit...")
  
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))
        start = time.time()
        outputs = self.session.run(None, {self.input_name: inputs})
        end = time.time()
        inference_time = end - start
        return np.squeeze(outputs).transpose((1,2,0)), inference_time

class ObjDetInferenceInstance():

    def __init__(self):
        self.od_handle = None
        self.cap_handle = None

    def create_objdet_handle(self, model_config_path):
        # Config file for Object Detection
        ret = os.path.exists('./model/model.config')

        # Check for model.config file
        if ret is False:
           print("\n ERROR: No model.config file found under model dir")
           print("\n Exiting inference....")
           sys.exit(0)

        self.od_handle = ONNXRuntimeObjectDetection("./model/model.config")

    def create_video_handle(self):

        # Currently supprots USB camera stream
        self.cap_handle = cv2.VideoCapture(0)
        if self.cap_handle is None:
           print("\n Error: Input Camera device not found/detected")
           print("\n Exisiting inference...")
           sys.exit(0)

    def model_inference(self):

        print("\n Loading model and labels file ")
        self.create_objdet_handle("./model/model.config")

        self.create_video_handle()

        # Reading widht and height details
        img_width = int(self.cap_handle.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(self.cap_handle.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Adding iot support
        # Choose HTTP, AMQP or MQTT as transport protocol.  Currently only MQTT is supported.
        IOT_HUB_PROTOCOL = IoTHubTransportProvider.MQTT
        iot_hub_manager = IotHubManager(IOT_HUB_PROTOCOL)

        while self.cap_handle.isOpened():
            # Caputre frame-by-frame
            ret, frame = self.cap_handle.read()
            predictions, infer_time = self.od_handle.predict_image(frame)

            for d in predictions:
                x = int(d['boundingBox']['left'] * img_width)
                y = int(d['boundingBox']['top'] * img_height)
                w = int(d['boundingBox']['width'] * img_width)
                h = int(d['boundingBox']['height'] * img_height)

                x_end = x+w
                y_end = y+h

                start = (x,y)
                end = (x_end,y_end)

                if 0.45 < d['probability']:
                    frame = cv2.rectangle(frame,start,end, (255, 255, 255), 1)

                    out_label = str(d['tagName'])
                    score = str(int(d['probability']*100))

                    cv2.putText(frame, out_label, (x-5, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, score, (x+w-50, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                    message = { "Label": out_label,
                                "Confidence": score,
                                "Position": [x, y, x_end, y_end],
                                "TimeStamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Send message to IoT Hub
                    if iot_hub_manager is not None:
                        iot_hub_manager.send_message_to_upstream(json.dumps(message))

            cv2.putText(frame, 'FPS: {}'.format(1.0/infer_time), (10,40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            if self.od_handle.disp == 1:
                # Displaying the image
                cv2.putText(frame, "Press 'r' on keyboard to refresh", (5,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Inference results", frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                   #break
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    #self.cap_handle.release()
                    #self.model_inference()
                    iot_hub_manager.restart_inferance(od_handle)
            else:
                for d in predictions:
                    print("Object(s) List: ", str(d['tagName']))

        # when everything done, release the capture
        self.cap_handle.release()
        cv2.destroyAllWindows()

def main():

    infer_obj = ObjDetInferenceInstance()

    print(" Starting model inference... ")
    infer_obj.model_inference()

if __name__ == '__main__':
    main()
