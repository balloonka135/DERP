#!/bin/bash

apt-get install gcc g++ python python-dev python-opencv python-pip
pip install -r requirements.txt
make all
make install
make clean