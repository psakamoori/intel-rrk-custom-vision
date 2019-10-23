## Azure ML Training and Deployement on Intel Edge device (RRK)

##Pre-requisites:
 - Intel powered Edge device (UP^2) with Ubuntu 16.04
 - USB webcam (/dev/video0)
 - OpenVINO installation with ORT (ONX Runtime) execution provider
 - Python OpenCV 

##Folders:
 #models (models.onnx, labesl.txt)
   - ONNX pre-trained models for Image_classificaiton
   - ONNX pre-trained models for object_detection
 #src
   - Python Application code for Image Classification and object detection
   - objdet/model.config - Model configuration file


##Ex:model.config (for Object detection)
 
  {
   "Network":0,  
   "modeltype":"onnx model",  
   "Input":"cam",    
   "display":1,  
   "Anchors": [[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]],  
   "ScaleWidth":416, 
   "ScaleHeight":416,  
   "InputFormat":"RGB",  
   "ConfThreshold":0.5,   
   "IOU_THRESHOLD":0.45,  
   "Runtime":1,  
   "MODEL_FILENAME":"../../models/object_detection/face_detect/model.onnx",  
   "LABELS_FILENAME":"../../models/object_detection/face_detect/labels.txt"  
  }
 
##Testing
 # Image-classifcation (with pre-trained ONNX models using customvision.ai)
   - Folder "models/image_classification" has few pre-trained onnx models
   - Folder "samples" has few example jpg's to use for testing
   - Initialize "model_path", "labels_path" and "image_path" accordingly in onnx_image_classification.py 
   - Execute command: src$ python3 onnx_image_classifciation.py
   - Expected Output: Predicted image classification result with label 

 # Object Detection (with pre-trained Face detection ONNX models using customvision.ai)
   - Folder "models/object_detect/" has face detection pre-trained onnx model
   - Execute command: src/objdet$ python3 onnxruntime_predict.py <model configuration file : model.config>
   - Expected Output: Renders webcam video frames with inference results (bounding box, detection label and score)
    ![](/output/objDet-FaceDetection.PNG) 
