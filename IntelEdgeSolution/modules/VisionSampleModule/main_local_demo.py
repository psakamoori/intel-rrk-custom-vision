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

import sys, os
import onnxruntime
import numpy as np
import cv2
import json
import time
import datetime
from store_to_blob import upload_to_cloud

from VideoStream import VideoStream
from predict import ObjectDetection
#from iot_hub_manager import IotHubManager
#from iothub_client import IoTHubTransportProvider, IoTHubError
from onnxruntime.capi.onnxruntime_pybind11_state import RunOptions

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
        self.img_width = 0
        self.img_height = 0
        self.cap_handle = None
        self.vs = None
        self.session = None
        self.cloudupload = False

        #onnxruntime.capi.onnxruntime_pybind11_state.RunOptions = False
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


    def create_video_handle(self):
        #usb_video_path = "rtsp://10.69.160.19:8900/live"
        usb_video_path = "./TF/testimages/OwnData/Gate2.m4v"

        self.vs = VideoStream(usb_video_path).start()

        # Reading widht and height details
        self.img_width = int(self.vs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(self.vs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def model_inference(self):

        self.create_video_handle()
        count=0
        upload_count=0
        while self.vs.stream.isOpened():
            count=count+1
            # Caputre frame-by-frame
            frame = self.vs.read()
            if(frame is None):
                print("cannot capture frame exiting")
        
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
                    out_label = str(d['tagName'])
                    if(out_label == 'car'):
                        cropframe = frame[y:y_end, x:x_end]
                        #cv2.imshow("car only cropped",cropframe)
                        
                        format = ".jpg"
                        filename = "car_" + str(upload_count) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + format
                        store_path =os.path.join(os.path.os.path.dirname(os.path.abspath(__file__)),"captured_images",filename)
                        #if(self.cloudupload):
                            #upload_to_cloud(cropframe,upload_count)
                        #else:
                            #cv2.imwrite(store_path,cropframe) 
                        upload_count += 1
                        if(full_frame == True):
                            frame = cv2.rectangle(frame,start,end, (255, 255, 255), 1)
                            score = str(int(d['probability']*100))
                            cv2.putText(frame, out_label, (x-5, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, score, (x+w-50, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.imwrite(store_path,frame)

                    frame = cv2.rectangle(frame,start,end, (255, 255, 255), 1)
                    score = str(int(d['probability']*100))

                    cv2.putText(frame, out_label, (x-5, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, score, (x+w-50, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                    message = { "Label": out_label,
                                "Confidence": score,
                                "Position": [x, y, x_end, y_end],
                                "TimeStamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Send message to IoT Hub
                    print(message)

            cv2.putText(frame, 'FPS: {}'.format(1.0/infer_time), (10,40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            if self.disp == 0:
                # Displaying the image
                cv2.imshow("Inference results", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   break

        # when everything done, release the capture
        self.vs.__exit__(None, None, None)
        cv2.destroyAllWindows()


def main():

    model_config_path = "./model/model.config"
    od_handle = ONNXRuntimeObjectDetection(model_config_path)

    # Adding iot support
    # Choose HTTP, AMQP or MQTT as transport protocol.  Currently only MQTT is supported.

    #print(" Starting model inference... ")
    od_handle.model_inference()

if __name__ == '__main__':
    main()
