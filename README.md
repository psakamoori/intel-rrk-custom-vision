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

##Testing
 # Image-classifcation
   - Folder "models/image_classification" has few pre-trained onnx models
   - Execute command: src/imgcls$ python3 onnx_image_classifciation.py  model.config
   - Expected Output: Predicted image classification result with label 
   -
     ##Example:model.config (for Image Classification)  
   {  
    "Network":0,  
    "modeltype":"onnx model",  
    "Input":"cam",  
    "display":1,  
    "mean_vec":[0.485, 0.456, 0.406],  
    "stddev_vec":[0.229, 0.224, 0.225],  
    "ScaleWidth":224,  
    "ScaleHeight":224,  
    "InputFormat":"RGB",  
    "Runtime":1,  
    "MODEL_FILENAME":"../../models/image_classification/dog_n_cat/model.onnx",  
    "LABELS_FILENAME":"../../models/image_classification/dog_n_cat/labels.txt"  
   }     
   Classification results  

   ![](/output/cat.png) 

 # Object Detection
   - Folder "models/object_detect/" has face detection pre-trained onnx model
   - Execute command: src/objdet$ python3 onnxruntime_predict.py model.config
   - Expected Output: Renders webcam video frames with inference results (bounding box, detection label and score)
   -
     ##Example:model.config (for Object detection)  
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

   Face Detection output 

   ![](/output/objDet-FaceDetection.png) 

   Car and Traffic Light Detection

   ![](/output/ObjDet-CarNTrafficLight.png)  

