# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which defined in object_detection.py DEFAULT_INPUT_SIZE)
import sys
import onnxruntime
import numpy as np
from PIL import Image, ImageDraw
from object_detection import ObjectDetection

MODEL_FILENAME = '../../models/object_detection/face_detect/model.onnx'
LABELS_FILENAME = '../../models/object_detection/face_detect/labels.txt'

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime
    """
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        self.session = onnxruntime.InferenceSession(model_filename)
        self.input_name = self.session.get_inputs()[0].name 
      
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0))

def main(image_filename):
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = ONNXRuntimeObjectDetection(MODEL_FILENAME, labels)

    image = Image.open(image_filename)
    predictions = od_model.predict_image(image)
    print(predictions)
    
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} image_filename'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
