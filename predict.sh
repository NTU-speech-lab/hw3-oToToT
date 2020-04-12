#!/bin/bash

# from https://github.com/NTU-speech-lab/hw3-oToToT/releases/tag/v0.0.6
gdown --id '1VvQonwOyuk1lTaJOoUx-Vdy6xPBfC80z' --output model.torch

python3 predict.py $1 $2
