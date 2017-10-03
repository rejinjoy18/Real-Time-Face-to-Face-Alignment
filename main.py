from __future__ import print_function

"""

This is the main module.
To call the program, execute python3 main.py <style_image> and
follow on screen directions for alignment and matting.

"""


import os 
import cv2
import sys
import numpy as np
import io
import dlib
from matplotlib import pyplot as plt


# Imports from other modules
from compute_matt import *
from align import *




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("style_image", help="path to style image")
    args = parser.parse_args()
    
    img_path = args.style_image
    
    
    inputt,unaligned, aligned = align(img_path)
    
    #matt_face = compute_matt(aligned)
    cv2.imwrite('/home//rejinjoy18//facealignment//mom1//aligned_warped.jpg', aligned)

    #matt_face_unaligned = compute_matt(unaligned)
    cv2.imwrite('/home//rejinjoy18//facealignment//mom1//aligned_unwarped.jpg', unaligned)

    
    cv2.imwrite('/home//rejinjoy18//facealignment//mom1//input.jpg', inputt)

    #matt_input = compute_matt(inputt)
    #cv2.imwrite('/home//rejinjoy18//facealignment//mom//input_m.jpg', matt_input)    
