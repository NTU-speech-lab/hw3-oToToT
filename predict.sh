#!/bin/bash

# from https://github.com/NTU-speech-lab/hw3-oToToT/releases/tag/v0.0.6
wget 'https://drive.google.com/uc?export=download&id=1VvQonwOyuk1lTaJOoUx-Vdy6xPBfC80z' -O model.torch

python3 predict.py $1 $2
