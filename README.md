## Azure ML Training and Deployement on Intel Edge device (RRK)

#Pre-requisites:
 - Intel powered Edge device - Ex: UP^2 with MyraidX with Ubuntu 16.04
 - OpenVINO installation with ORT (ONX Runtime) execution provider

#Testing
 ## Image-classifcation (with pre-trained ONNX models using customvision.ai)
   - Folder "models/image_classification" has few pre-trained models for image classifcation
   - Folder "samples" has few example jpg's to use for testing
   - Initialize "model_path", "labels_path" and "image_path" accordingly in onnx_image_classification.py and run below command for project root directory
     $python3 onnx_image_classifciation.py
   - Expected Output: Predicted image classification result with label 

 ## Object Detection (with pre-trained Face detection ONNX models using customvision.ai)
   - Folder "models/object_detect/" has few pre-trained models for image classifcation
   - Folder "samples" has few example jpg's to use for testing (face.jpg - pass it as command argument)
   - Initialize "model_path", "labels_path" and accordingly in onnxruntime_predict.py and run below command for project root directory
     src/objdet$ python3 onnxruntime_predict.py ../../samples/face.jpg
   - Expected Output: List of dictonaries with Key (predicted object) and Values (bounding box values)
 
