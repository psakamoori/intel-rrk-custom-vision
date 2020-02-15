
Deploying OpenVINO: AI Vision Module onto IoT Edge device
======================

**Supported Platforms**
===========
1. Hardware Used:
      CPU (Host Processor): Intel® Core™ i7-7567U CPU @ 3.5GHz × 4  
      VPU (Accelerator): Intel® Neural Compute Stick 2 (NCS2)
2. Operating System: Ubuntu 16.04


**Pre-requisites**
==============

1. Azure account 
    - Access to portal.azure.com
    - Access to customvision.ai
2. Docker Installation
   - Docker : https://docs.docker.com/install/linux/docker-ce/ubuntu/
3. Installations
   - Intel® Distribution of OpenVINO™ toolkit: Download and install OpenVINO 2019_R3 from [link](http://docs.openvinotoolkit.org/2019_R3/_docs_install_guides_installing_openvino_linux.html)
   - Drivers for NCS2: Intel® Vision Accelerator Design with Intel® Movidius™ VPUs on Linux* using [link](https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick)
   - Azure IoT Edge Runtime [link](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge-linux)
4. Enabling xhost on IoT Edge device:
   - Linux terminal command: $xhost +SI:localuser:root
   
**Hardware Accessories and Enabling xhost**
===============
  1. Ensure that the USB Camera is connected to edge device.
  2. Ensure that the Intel® Neural Compute Stick 2 is connected to the edge device.

**Starting inference with object detection or image classification ONNX model (from customvision.ai zip url)**
==================================
Follow video and/or snapshots on Azure Marketplace

Tried it? Join the conversation on the [Community Forum](https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit)

**Troubleshooting**
================

[1] USB Device not found (for NCS2)

Container Create Options to mount NCS2 inside docker:

   {
    "HostConfig": {
        "Binds": [
            "/tmp/.X11-unix:/tmp/.X11-unix",
            "/dev:/dev"
        ],
        "NetworkMode": "host",
        "IpcMode": "host",
        "Privileged": true
    },
    "NetworkingConfig": {
        "EndpointsConfig": {
            "host": {}
        }
    },
    "Env": [
        "DISPLAY=:0"
    ]
  } 

  Once app is deployed on to Edge device follow below steps

   1. Go to Azure IoT Hub on your portal.azure.com
   2. Select "IoT Edge" under "Automatic Device Managment"
   3. Select IoT device 
   4. Click "Set Modules"
   5. Click on Edge module w.r.t application deployed
   6. Select "Container Create Options" and past above code in there and click "Update"
   7. Finally "Review+Create"
   8. Restart your Edge module "sudo iotedge restart <module name>"
