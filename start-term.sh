#!/bin/bash

conda init bash
source ~/.bashrc
conda activate /home/jovyan/omero-env
sudo apt-get update
#sudo apt-get install -y libvips
sudo apt-get install -y libgl1
#sudo apt-get install -y git