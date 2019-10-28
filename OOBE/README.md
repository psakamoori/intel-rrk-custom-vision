
Instructions to install Azure IoT Edge Runtime and Azure IoT Resoruce creation

## Pre-requisites 
  - Azure Cloud Account
  - Edge device (Ex:UP^2)  with Ubuntu 16.04
  
## Install Azure IoT Edge Runtime & dependencies

  - Execute below script on Edge device 
    - $sudo ./az-iot-edge-runtime-install.sh 

## Create Azure IoT Resoeruces

  - Execute below script to setup IoT Resources (IoT Hub, IoT Edge & Generative device connection string)
    -$sudo python3 az-resoruce-create.py <Azure account user-name>
  - User input: Password to Azure Cloud Account
