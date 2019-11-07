# ==============================================================================
# Copyright (c) Microsoft Corporation. All rights reserved.
# 
# Licensed under the MIT License.
# ==============================================================================

import time
import os
import subprocess as sp
import sys
import shutil
import socket
import logging
import json
import urllib.request as urllib2
from urllib.request import urlopen
import glob
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.disabled = False


def WaitForFileDownload(FileName):
    # ----------------------------------------------------
    # Wait until the end of the download
    # ----------------------------------------------------
    valid=0
    while valid==0:
        try:
            with open(FileName):valid=1
        except IOError:
            time.sleep(1)
    print("Got it ! File Download Complete !")

def get_file_zip(url,dst_folder="model") :
    #adding code to fix issue where the file name may not be part of url details here 
    #
    remotefile = urlopen(url)
    myurl = remotefile.url
    FileName = myurl.split("/")[-1]
    if FileName:
        # find root folders
        dirpath = os.getcwd()
        dirpath_file = os.path.join(dirpath,dst_folder)
        src = os.path.abspath(dirpath_file)
        src_file_path = os.path.join(src,FileName)
        logger.info("location to download is ::" + src_file_path)
        prepare_folder(dirpath_file)
        print("Downloading File ::" + FileName)

        urllib2.urlretrieve(url, filename=src_file_path)
        WaitForFileDownload(src_file_path)
        result=unzip_and_move(src_file_path)

        return result
    else:
        print("Cannot extract file name from URL")
        return False

def unzip_and_move(file_path=None,):
    zip_ref = zipfile.ZipFile(file_path,'r')
    dirpath = os.getcwd()
    dirpath_file = os.path.join(dirpath,"model")
    zip_ref.extractall(dirpath_file)
    zip_ref.close()
    logger.info("files unzipped to : " + dirpath_file)
    return True


#if __name__ == "__main__":
     #get_file_zip("https://yadavsrorageaccount01.blob.core.windows.net/visionstoragecontainer/a5719e7549c044fcaf83381a22e3d0b2.VAIDK.zip","twin_provided_model")




