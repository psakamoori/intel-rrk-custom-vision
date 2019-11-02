#!/bin/bash
#!/usr/bin/python3

# Copyright (c) 2019 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Purpose: Check/Create Azure IoT resources
#
#    - Login into Azure IoT account
#    - check/Create Azure IoT Resources
#    - Check/Create Azure IoT Hub
#    - Check/Create Azure IoT Edge
#    - Extract Azure IoT Edge device connection string
#
# Usage:
#    - $ sudo python3 ./az-resoruce-create.py <azure-account-user-login>
#
# Note: Currently script checks for resoruce present or not
#       if present it will use them (wont support creating new, if present)
#
# Author: Pradeep, Sakhamoori <pradeep.sakhamoori@intel.com>
# Date  : 10/28/2019
#

import sys
import os, json
import subprocess

from getpass import getpass

global sub_id_name
global rs_grp_name, hub_name, dev_name

def main(username):
    global sub_id_name
    passwd = getpass("Enter Azure account login password: ")
    #subprocess.call(['az', 'login', '-u', username, '-p', passwd])
    proc = subprocess.Popen(['az', 'login', '-u', username, '-p', passwd],
                                                stdout=subprocess.PIPE)
    (sub_id, err) = proc.communicate()
    sub_id = sub_id.decode('utf-8')
    sub_id_ = json.loads(sub_id)
    sub_id_name = sub_id_[0]['id']

    #sub_id_name = sub_id_name.split('/')
    #print("sub_id_name", sub_id_name)

    check_for_res_group()

def check_for_res_group():
   global rs_grp_name
   # Check if resoruce group exists; if not create one
   proc = subprocess.Popen(['az', 'group', 'list'], 
                               stdout=subprocess.PIPE)

   (rs_group, err) = proc.communicate()
   rs_group = rs_group.decode('utf-8')

   # check length of rs_group array; empty array length will be 1
   if len(rs_group) > 3:
      rs_grp = json.loads(rs_group)
      rs_grp_name = rs_grp[0]['name']
      print("\n== Resource Group Found ==")
      print("Resoruce Group INFO = ", rs_group)
      print("Resource Group NAME = ", rs_grp[0]['name'])
   else:
      print("\n ***Warning: No Resource Found\n")
      print("\n Creating Resoruce Group")
      grp_name = input("\n Enter Group Name: ")
      rs_grp_name = grp_name     
      reg_name = input("\n Enter Region Name: ")
      subprocess.call(['az', 'group', 'create', '--name', grp_name,
                       '--location', reg_name])
   check_for_hub()

def check_for_hub():
    global rs_grp_name, hub_name
    #Check if IoT Hub exists; if not create one
    proc = subprocess.Popen(['az', 'iot', 'hub', 'list'],
                            stdout=subprocess.PIPE)

    (iot_hub, err) = proc.communicate()
    iot_hub = iot_hub.decode('utf-8')

    if len(iot_hub) > 3:
       iot_hub_ = json.loads(iot_hub)
       hub_name = iot_hub_[0]['name']
       print("\n== IoT Hub Found ==")
       print("\nIoT Hub Name: ", hub_name)
    else:
       print("\n ***Warning: No IoT hub found")
       print("\n Creating IoT Hub")
       hub_name=input("\n Enter IoT Hub name: ")
       subprocess.call(['az', 'iot', 'hub', 'create', '--resource-group',
                         rs_grp_name, '--name', hub_name])
      
       print("\n IoT Hub " + str(hub_name) + " Created")

    check_iot_edge()

def check_iot_edge():
    global hub_name, dev_name
 
    print("\n Checking if IoT Hub ", str(hub_name), "exisits")

    # Adding Azure CLI extension
    subprocess.call(['az', 'extension', 'add', '--name', 'azure-cli-iot-ext'])
    
    # Check if IoT Edge present under IoT Hub
    proc = subprocess.Popen(['az', 'iot', 'hub', 'device-identity', 'list',
                             '--hub-name', str(hub_name)],
                             stdout=subprocess.PIPE)

    (dev_id, err) = proc.communicate()
    dev_id = dev_id.decode('utf-8')
    print("\n hub_dev_id", dev_id)

    if len(dev_id) > 3:
       print("\n == IoT Edge device found== ")
       dev_id_ = json.loads(dev_id)
       #print("dev_id_", dev_id_)
       dev_name = dev_id_[0]['deviceId']
       print("\n IoT Edge device: ", dev_name)
    else:
       print("\n*** Warning: IoT Edge deivce not Found")
       print("\n Creating IoT Edge Device")

       dev_name = input("\n Enter IoT Edge device name: ")
       subprocess.call(['az', 'iot', 'hub', 'device-identity', 'create',
                        '--device-id', dev_name, '--hub-name', hub_name,
                        '--edge-enabled'])

       print("\n IoT Edge " + str(dev_name) + " Created")
    
    get_device_string()

def get_device_string():
    global hub_name, dev_name

    print("\n Retrieve device connection string")
    proc = subprocess.Popen(['az', 'iot', 'hub', 'device-identity',
                             'show-connection-string', '--device-id',
                              str(dev_name), '--hub-name', str(hub_name)],
                              stdout=subprocess.PIPE)

    (dev_str, err) = proc.communicate()
    dev_str = dev_str.decode('utf-8')
    
    print("\n dev_str: ", dev_str)    
    print("\n USE DEVICE CONNECTION STRING TO UPDATE-/etc/iotedge/config.yaml")

    check_for_acr()

def check_for_acr():
    global sub_id_name, rs_grp_name

    print("\n Check for ACR - Azure Container Registry")
    proc = subprocess.Popen(['az', 'acr', 'list', '--resource-group', str(rs_grp_name),
                     '--subscription', str(sub_id_name)], stdout=subprocess.PIPE)

    (az_acr, err) = proc.communicate()
    az_acr = az_acr.decode('utf-8')

    print("az_acr", az_acr)

    if len(az_acr) > 3:
       print("\n == Azure ACR found == ")
       az_acr_ = json.loads(az_acr)
       #print("az_acr_", az_acr_)

       acr_name = az_acr_[0]['name']
       print("\n ACR name: ", acr_name)
    else:
       print("\n** Warning: AZure ACR not found ")
       print("\n Creating Azure ACR ")

       az_acr_name = input("\n Enter Azure ACR name : ")

       subprocess.call(['az', 'acr', 'create', '-n', az_acr_name, '-g',
                        rs_grp_name, '--sku', "standard"])

       print("\n Azure ACR " + str(az_acr_name) + " Created ")

    print("\n IoT Resoruces ready to start deploying AI on Edge")

if __name__ == '__main__':
   if len(sys.argv) <= 1:
      print('USAGE: {} <Azure account username>'.format(sys.argv[0]))
   else:
      main(sys.argv[1])

