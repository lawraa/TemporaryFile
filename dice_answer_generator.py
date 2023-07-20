import cv2
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy
import os
import json

def gener(file_name):
    # Get the current working directory
    current_dir = os.getcwd()

    # Navigate to the desired directory
    os.chdir("../")
    os.chdir("upload/DataToImage/ds-output/mnist2/Test")
    

    # Open the file in read mode
    with open(file_name, 'r') as json_file:
    # Load the JSON data
        json_data = json.load(json_file)

    segmentation = []
#------------------------------------    change this    ----------------------------------------------------------
    if json_data['annotation']:
        # Check if 'segmentation' key is present
        if 'segmentation' in json_data['annotation'][0]:
            segmentation = json_data['annotation'][0]['segmentation']
            #print(segmentation)


    coords = []
    
    for i in range(len(segmentation)):
        coords.append(list(zip(segmentation[i][0::2], segmentation[i][1::2])))

    
    os.chdir(current_dir)
    imagee = Image.open("test_predict_output_demo/all_black.png")
    draw = ImageDraw.Draw(imagee)

    # Draw the polygons on the image    
    for i in range (len(coords)):
        draw.polygon(coords[i], fill="white")
    
#------------------------------------    change this    ----------------------------------------------------------
    name = file_name
    name = name.replace(".json", "")
    name = "test_predict_output/"+name+"_test.png"

    # Save the modified image
    imagee.save(name)

            
            
