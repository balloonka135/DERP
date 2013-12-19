#!/bin/bash

cp config.py.default config.py
sudo apt-get install gcc g++ python python-dev python-opencv python-pip
sudo pip install -r requirements.txt
make all
sudo make install
make clean