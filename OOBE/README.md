
Instructions to install Azure IoT Edge Runtime and Azure IoT Resoruce creation

## Pre-requisites 
  - Azure Cloud Account
  - Edge device (Ex:UP^2)  with Ubuntu 16.04
  
## Step 1: Install Azure IoT Edge Runtime & dependencies

  - Execute below script on Edge device 
    - $sudo ./az-iot-edge-runtime-install.sh 

## Step 2: Create Azure IoT Resoeruces

  - Execute below script to setup IoT Resources (IoT Hub, IoT Edge & Generative device connection string)
    -$sudo python3 az-resoruce-create.py <Azure account user-name>
  - User input: Password to Azure Cloud Account
  - Output:
     - Resoruce Group Info
     - IoT Hub Info
     - Device connection string 

## Step 3: Update device connection string

  - Copy "device connection string" from Step 2 to "sudo nano /etc/iotedge/config.yaml"
     - Replace this "<ADD DEVICE CONNECTION STRING HERE"> with device string looks like below 
       Ex: "HostName=devinwonIoTHub.azure-devices.net;DeviceId=AIVisionDemo1;SharedAccessKey=DBnciasdasdasdasd/7iRasdasfasfoxasmqhtqa7X0nTk7o=""

