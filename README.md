## Azure ML Training and Deployement on Intel Edge device (RRK)

#Pre-requisites:
 - Intel powered Edge device - Ex: UP^2 with MyraidX with Ubuntu 16.04
 - OpenVINO installation with ORT (ONX Runtime) execution provider

#Steps to test image-classifcation inference (with pre-trained ONNX models using customvision.ai)
 - Folder "models/image_classification" has few pre-trained models for image classifcation
 - Folder "samples" has few example jpg's to use for testing
 - Initialize "model_path", "labels_path" and "image_path" accordingly in onnx_image_classification.py and run below command for project root directory
   $python3 onnx_image_classifciation.py

# Expected output of Image Classification
 - Inference results will display confidence score of image classification with label info
 
