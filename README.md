## Azure ML Training and Deployement on Intel Edge device (RRK)

#Pre-requisites:
 - Intel powered Edge device - Ex: UP^2 with MyraidX with Ubuntu 16.04
 - OpenVINO installation with ORT (ONX Runtime) execution provider

#Testing
 ## Image-classifcation (with pre-trained ONNX models using customvision.ai)
   - Folder "models/image_classification" has few pre-trained onnx models
   - Folder "samples" has few example jpg's to use for testing
   - Initialize "model_path", "labels_path" and "image_path" accordingly in onnx_image_classification.py 
   - Execute command: src$ python3 onnx_image_classifciation.py
   - Expected Output: Predicted image classification result with label 

 ## Object Detection (with pre-trained Face detection ONNX models using customvision.ai)
   - Folder "models/object_detect/" has face detection pre-trained onnx model
   - Folder "samples" has few example jpg's to use for testing (face.jpg - pass it as command argument)
   - Initialize "model_path", "labels_path" and accordingly in onnxruntime_predict.py 
   - Execute command: src/objdet$ python3 onnxruntime_predict.py ../../samples/face.jpg
   - Expected Output: List of dictonaries with Key (predicted object) and Values (bounding box values)
 
