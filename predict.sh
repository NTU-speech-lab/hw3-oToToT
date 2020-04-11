#!/bin/bash

wget 'https://ototot.tk/cloud/ML/hw3.torch' -O model.torch

python3 predict.py $1 $2
