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
import datetime

from getpass import getpass

class iot_edge_resoruce_create():
    def __init__(self):
        self.rs_grp_name = None
        self.hub_name = None
        self.dev_name = None
        self.dev_str = None
        self.sub_id_name = None
        self.az_acr_name = None

    def azure_ac_login(self, username):
        passwd = getpass("Enter Azure account login password: ")
        #subprocess.call(['az', 'login', '-u', username, '-p', passwd])
        proc = subprocess.Popen(['az', 'login', '-u', username, '-p', passwd],
                                              stdout=subprocess.PIPE)
        (sub_id, err) = proc.communicate()
        sub_id = sub_id.decode('utf-8')
        sub_id_ = json.loads(sub_id)
        self.sub_id_name = sub_id_[0]['id']

        self.check_for_res_group()

    def create_new_resource_group(self):

        self.logs(" Creating Resoruce Group ")
        self.rs_grp_name = input("\n Enter Group Name: ")
        reg_name = input("\n Enter Region Name: ")
        subprocess.call(['az', 'group', 'create', '--name', self.rs_grp_name,
                           '--location', reg_name])

    def check_for_res_group(self):
        # Check if resoruce group exists; if not create one
        proc = subprocess.Popen(['az', 'group', 'list'], 
                                   stdout=subprocess.PIPE)

        (rs_group, err) = proc.communicate()
        rs_group = rs_group.decode('utf-8')

        # check length of rs_group array; empty array length will be 1
        if len(rs_group) > 3:
           rs_gr_lst = []
           rs_grp = json.loads(rs_group)
           self.logs("No.of. RGs found: " + str(len(rs_grp)))

           for i in range(len(rs_grp)):
               #self.rs_grp_name.append(str(rs_grp[i]['name']))
               rs_gr_lst.append(str(rs_grp[i]['name']))
               #self.logs("RG Info = " + str(rs_group))
               self.logs("RG Num and Name = " + str(i) + " " + str(rs_grp[i]['name']))
           self.logs(" WANT TO CREATE NEW RESOURCE GROUP [yes/no]")
           new_rs_grp_flag = input()
           if new_rs_grp_flag == 'yes':
              self.create_new_resource_group()
           if new_rs_grp_flag == 'no':
              self.logs("Select Resource Group from below list: \n" + str(rs_gr_lst))
              self.rs_grp_name = input()
              if str(self.rs_grp_name) not in rs_gr_lst:
                 self.log("WARNING: In-correct resource group name selected")
                 self.rs_grp_name = None
        else:
           self.logs(" ***Warning: No Resource Found ")
           self.create_new_resource_group()

        self.check_for_hub()
 
    def create_new_iot_hub(self):
        self.logs(" Creating IoT Hub ")
        self.hub_name=input("\n Enter IoT Hub name: ")
        subprocess.call(['az', 'iot', 'hub', 'create', '--resource-group',
                         self.rs_grp_name, '--name', self.hub_name])

        self.logs(" IoT Hub " + str(self.hub_name) + " Created")


    def check_for_hub(self):
        #Check if IoT Hub exists; if not create one
        #proc = subprocess.Popen(['az', 'iot', 'hub', 'list'],
        #                         stdout=subprocess.PIPE)

        proc = subprocess.Popen(['az', 'iot', 'hub', 'list', '--resource-group', self.rs_grp_name],
                                 stdout=subprocess.PIPE)

        (iot_hub, err) = proc.communicate()
        iot_hub = iot_hub.decode('utf-8')

        if len(iot_hub) > 3:
           iot_hub_ = json.loads(iot_hub)
           self.hub_name = iot_hub_[0]['name']
           self.logs(" IoT Hub Found ")
           self.logs(" IoT Hub Name: " + str(self.hub_name))
           self.logs(" WANT TO CREATE NEW IoT Hub [yes/no]")
           new_iot_hub_flag = input()
           if new_iot_hub_flag == 'yes':
              self.create_new_iot_hub()
        else:
           self.logs(" ***Warning: No IoT hub found ")
           self.create_new_iot_hub()

        self.check_iot_edge()

    def create_new_iot_edge_device(self):
        self.logs(" Creating IoT Edge Device ")

        self.dev_name = input("\n Enter IoT Edge device name: ")
        subprocess.call(['az', 'iot', 'hub', 'device-identity', 'create',
                     '--device-id', self.dev_name, '--hub-name', self.hub_name,
                     '--edge-enabled'])

        self.logs(" IoT Edge " + str(self.dev_name) + " created ")


    def check_iot_edge(self):
 
        self.logs(" Checking if IoT Hub " + str(self.hub_name) + " present ")

        # Adding Azure CLI extension
        subprocess.call(['az', 'extension', 'add', '--name', 'azure-cli-iot-ext'])

        # Check if IoT Edge present under IoT Hub
        proc = subprocess.Popen(['az', 'iot', 'hub', 'device-identity', 'list',
                                 '--hub-name', str(self.hub_name)], stdout=subprocess.PIPE)

        (dev_id, err) = proc.communicate()
        dev_id = dev_id.decode('utf-8')
        self.logs(" hub_dev_id" + str(dev_id))

        if len(dev_id) > 3:
           self.logs(" IoT Edge device found ")
           dev_id_ = json.loads(dev_id)
           self.dev_name = dev_id_[0]['deviceId']
           self.logs(" IoT Edge device: " + str(self.dev_name))
           self.logs(" WANT TO CREATE NEW IoT EDGE DEVICE [yes/no]")
           new_edge_flag = input()
           if new_edge_flag == 'yes':
              self.create_new_iot_edge_device()
        else:
           self.logs(" *** Warning: IoT Edge deivce not found ")
           self.create_new_iot_edge_device()

        self.get_device_string()

    def get_device_string(self):

        self.logs(" Retrieve device connection string ")
        proc = subprocess.Popen(['az', 'iot', 'hub', 'device-identity',
                                'show-connection-string', '--device-id',
                                 str(self.dev_name), '--hub-name', str(self.hub_name)],
                                 stdout=subprocess.PIPE)

        (dev_str, err) = proc.communicate()
        self.dev_str = dev_str.decode('utf-8')
    
        self.logs(" dev_str: " + str(self.dev_str))
        self.logs(" USE DEVICE CONNECTION STRING TO UPDATE-/etc/iotedge/config.yaml")

        self.check_for_acr()

    def create_new_acr(self):

        self.logs("Creating Azure ACR ")

        self.az_acr_name = input("\n Enter Azure ACR name : ")
        subprocess.call(['az', 'acr', 'create', '-n', self.az_acr_name,
                         '-g', self.rs_grp_name, '--sku', "standard"])

        self.logs("Azure ACR " + str(self.az_acr_name) + " created ")
        self.logs("IoT Resoruces ready to start deploying AI on Edge")

    def check_for_acr(self):

        self.logs("Check for ACR - Azure Container Registry ")
        proc = subprocess.Popen(['az', 'acr', 'list', '--resource-group', 
                                str(self.rs_grp_name), '--subscription', 
                                str(self.sub_id_name)], stdout=subprocess.PIPE)

        (az_acr, err) = proc.communicate()
        az_acr = az_acr.decode('utf-8')

        self.logs("az_acr" + str(az_acr))
        new_acr_flag = None

        if len(az_acr) > 3:
           self.logs(" Azure ACR found ")
           az_acr_ = json.loads(az_acr)
           self.az_acr_name = az_acr_[0]['name']
           self.logs(" ACR name: " + str(self.az_acr_name))
           self.logs(" WANT TO CREATE NEW ACR [yes/no]")
           new_acr_flag = input()
           if new_acr_flag == 'yes':
              self.create_new_acr()
        else:
           self.logs("** Warning: AZure ACR not found ")
           self.create_new_acr()

    def logs(self, txt):
        print("["+ str(datetime.datetime.now()) + "]: " + str(txt))

if __name__ == '__main__':
   if len(sys.argv) <= 1:
      print('USAGE: {} <Azure account username>'.format(sys.argv[0]))
   else:
      iot_ob = iot_edge_resoruce_create()
      iot_ob.azure_ac_login(sys.argv[1])

      #main(sys.argv[1])

