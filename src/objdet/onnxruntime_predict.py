# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which defined in object_detection.py DEFAULT_INPUT_SIZE)
import sys
import onnxruntime
import numpy as np
import cv2
import json
from PIL import Image, ImageDraw
from object_detection import ObjectDetection

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
        self.session = onnxruntime.InferenceSession(self.model_filename)
        self.input_name = self.session.get_inputs()[0].name 
      
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0))

def main(config_filename):
    od_model = ONNXRuntimeObjectDetection(config_filename)

    if od_model.video_inp == 'cam':
        cap = cv2.VideoCapture(0)
        # Reading widht and height details
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if cap == None:
           print("Error: Input Camera device not found/detected")
    else:
        print("Error: Invalid input argument/source")

    color = (255, 0 , 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1

    while(True):
       # Caputre frame-by-frame
       ret, frame = cap.read()
       #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       predictions = od_model.predict_image(frame)

       for d in predictions:
           x = int(d['boundingBox']['left'] * img_width)
           y = int(d['boundingBox']['top'] * img_height)
           w = int(d['boundingBox']['width'] * img_width)
           h = int(d['boundingBox']['height'] * img_height)

           x_end = x+w
           y_end = y+h

           start = (x,y)
           end = (x_end,y_end)

           frame = cv2.rectangle(frame,start,end,color,thickness)
           out_label = str(d['tagName'])
           score = str(int(d['probability']*100))
           frame = cv2.putText(frame, out_label, (x-5, y), font,
                   fontScale, color, thickness, cv2.LINE_AA)
           frame = cv2.putText(frame, score, (x+w-50, y), font,
                   fontScale, color, thickness, cv2.LINE_AA)
           #print("class -", out_label)

       if od_model.disp == 1:
           # Displaying the image
           cv2.imshow("Inference results", frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
          break

       #input("Press Enter to continue...")
       #print(predictions)
    
    # when everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} config_filename'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
