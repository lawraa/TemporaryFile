# -*- coding: utf-8 -*-
import sys
import io
import requests
import json
import base64
from PIL import Image
import numpy as np
import gradio as gr
import os
import pandas as pd


sys.path.append('.')

max_imagebox = 10

saved_prompts = []

title = "SegGPT: Segmenting Everything In Context<br> \
<div align='center'> \
<h2><a href='https://arxiv.org/abs/2304.03284' target='_blank' rel='noopener'>[paper]</a> \
<a href='https://github.com/baaivision/Painter' target='_blank' rel='noopener'>[code]</a></h2> \
<br> \
<image src='file/rainbow.gif' width='720px' /> \
<h2>SegGPT performs arbitrary segmentation tasks in images or videos via in-context inference, such as object instance, stuff, part, contour, and text, with only one single model.</h2> \
</div> \
"

def inference_mask1_sam(prompt,img, img_):
    files = {
        "useSam" : 1,
        "pimage" : resizeImg(prompt["image"]),
        "pmask" : resizeImg(prompt["mask"]),
        "img" : resizeImg(img),
        "img_" : resizeImg(img_)
    }
    r = requests.post("http://120.92.79.209/painter/run", json = files)
    a = json.loads(r.text)

    res = []
    for i in range(len(a)):
        res.append(np.uint8(np.array(Image.open(io.BytesIO(base64.b64decode(a[i]))))))

    return res[1:] # remove the prompt image

def resizeImg(img):
    res, hres = 448, 448
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    temp = io.BytesIO()
    img.save(temp, format="WEBP")
    return base64.b64encode(temp.getvalue()).decode('ascii')

def resizeImgIo(img):
    res, hres = 448, 448
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    temp = io.BytesIO()
    img.save(temp, format="WEBP")
    return io.BytesIO(temp.getvalue())


# define app features and run

examples = [
            ['./images/hmbb_1.jpg', './images/hmbb_2.jpg'],
            ['./images/rainbow_1.jpg', './images/rainbow_2.jpg'],
            ['./images/earth_1.jpg', './images/earth_2.jpg'],
            ['./images/obj_1.jpg', './images/obj_2.jpg'],
            ['./images/ydt_2.jpg', './images/ydt_1.jpg'],
           ]
examples_pred = [
            ['./images/hmbb_3.jpg'],
            ['./images/rainbow_3.jpg'],
            ['./images/earth_3.jpg'],
            ['./images/obj_3.jpg'],
            ['./images/ydt_3.jpg'],
           ]
'''
examples = [
            ['./images/hmbb_1.jpg', './images/hmbb_2.jpg', './images/hmbb_3.jpg'],
            ['./images/rainbow_1.jpg', './images/rainbow_2.jpg', './images/rainbow_3.jpg'],
            ['./images/earth_1.jpg', './images/earth_2.jpg', './images/earth_3.jpg'],
            ['./images/obj_1.jpg', './images/obj_2.jpg', './images/obj_3.jpg'],
            ['./images/ydt_2.jpg', './images/ydt_1.jpg', './images/ydt_3.jpg'],
           ]
'''
examples_video = [
            ['./videos/horse-running.jpg', './videos/horse-running.mp4'],
            ['./videos/a_man_is_surfing_3_30.jpg', './videos/a_man_is_surfing_3_30.mp4'],
    ['./videos/a_car_is_moving_on_the_road_40.jpg', './videos/a_car_is_moving_on_the_road_40.mp4'],
['./videos/jeep-moving.jpg', './videos/jeep-moving.mp4'],
['./videos/child-riding_lego.jpg', './videos/child-riding_lego.mp4'],
['./videos/a_man_in_parkour_100.jpg', './videos/a_man_in_parkour_100.mp4'],
]


def demo_function2(prompt1,prompt2,prompt3,prompt4,prompt5,prompt6,prompt7,prompt8,prompt9,prompt10,
                 img):
    prompt_image_list = []
    prompt_tgt_list = []
    if prompt1:
        prompt_image_list.append(resizeImg(prompt1["image"]))
        prompt_tgt_list.append(resizeImg(prompt1["mask"]))
    if prompt2:
        prompt_image_list.append(resizeImg(prompt2["image"]))
        prompt_tgt_list.append(resizeImg(prompt2["mask"]))
    if prompt3:
        prompt_image_list.append(resizeImg(prompt3["image"]))
        prompt_tgt_list.append(resizeImg(prompt3["mask"]))
    if prompt4:
        prompt_image_list.append(resizeImg(prompt4["image"]))
        prompt_tgt_list.append(resizeImg(prompt4["mask"]))
    if prompt5:
        prompt_image_list.append(resizeImg(prompt5["image"]))
        prompt_tgt_list.append(resizeImg(prompt5["mask"]))
    if prompt6:
        prompt_image_list.append(resizeImg(prompt6["image"]))
        prompt_tgt_list.append(resizeImg(prompt6["mask"]))
    if prompt7:
        prompt_image_list.append(resizeImg(prompt7["image"]))
        prompt_tgt_list.append(resizeImg(prompt7["mask"]))
    if prompt8:
        prompt_image_list.append(resizeImg(prompt8["image"]))
        prompt_tgt_list.append(resizeImg(prompt8["mask"]))
    if prompt9:
        prompt_image_list.append(resizeImg(prompt9["image"]))
        prompt_tgt_list.append(resizeImg(prompt9["mask"]))        
    if prompt10:
        prompt_image_list.append(resizeImg(prompt10["image"]))
        prompt_tgt_list.append(resizeImg(prompt10["mask"]))   
    
    x= 0.488

    return img,x



def demo_function(prompts, img):
    prompt_image_list = []
    prompt_tgt_list = []
    
    for prompt in prompts:
        if prompt is not None:
            prompt_image_list.append(resizeImg(prompt["image"]))
            prompt_tgt_list.append(resizeImg(prompt["mask"]))
    
    # Process the prompts and final image here and return the output image
    return img



def save_prompt_handler(prompts):
    global saved_prompts
    saved_prompts.append(version)

def load_prompt_handler(version):
    global saved_prompts
    if version < len(saved_prompts):
        return saved_prompts[version]
    else:
        return None

def variable_outputs(x):
    x = int(x)
    return [gr.ImageMask.update(visible=True)]*x+ [gr.ImageMask.update(value= None, visible=False)]*(max_imagebox-x)


with gr.Blocks() as demo:
    gr.Markdown(
        #title
        """
        # Demo
        Scroll for amount of input prompt
        Predict Image at the right
        """
    )
    with gr.Tab("Prompting and Predicting"):
        with gr.Row():
            with gr.Column():
                slider = gr.Slider(1,max_imagebox, step = 1, value = 10, label = "Amount of Prompts: ")
                prompt_button = gr.Button(value = "Submit Amount of Prompt").style()  
        with gr.Row():
            gr.Markdown(
                """
                # Upload and Predict
                Upload your Prompt on the Left
                Predict Image on the right
                """
            )
        with gr.Row():
            with gr.Column():
                imagebox = []
                for i in range(max_imagebox):
                    with gr.Row():
                        masked_image = gr.ImageMask(brush_radius=8, label = "Prompt").style(height = 350)
                        imagebox.append(masked_image)
                        
            with gr.Column():
                with gr.Row():
                    predict_img = gr.Image(shape=(240, 240),label = "Image", height = 350)
                with gr.Row():
                    
                    clear_prompt_button = gr.ClearButton(imagebox, value = 'Clear All Prompt')
                    clear_image_button = gr.ClearButton(predict_img, value = 'Clear Predict Image')
                with gr.Row():
                    predict_button = gr.Button('Predict')
                with gr.Row():
                    save_button = gr.Button('Save Prompts')
                    load_button = gr.Button('Load Prompts')
                with gr.Row():
                    with gr.Column():
                        prediction_result = gr.Image(label = "Prediction Result", height = 350)
                        performance = gr.Textbox(label = "Performance(Dice Coefficient)")




#SLIDE THEN CLICK TO CHANGE
            prompt_button.click(variable_outputs, slider, imagebox)

#SLIDE AND CHANGE IMMEDIATELY
            #slider.change(variable_outputs, slider, imagebox)


            print("HI")
            print(imagebox)
            #predict_button.click(demo_function, inputs= [imagebox, predict_img], outputs=prediction_result)
            predict_button.click(demo_function2, inputs= [imagebox[0], imagebox[1],imagebox[2],imagebox[3],imagebox[4],imagebox[5],imagebox[6],imagebox[7],imagebox[8],imagebox[9], predict_img], outputs=[prediction_result, performance])
            #load_button.click()


        with gr.Row():
            with gr.Column():
                gr.Markdown("## Test Examples")
                gr.Examples(
                    examples = examples,
                    inputs = imagebox,
                    cache_examples = False,
                )
                gr.Markdown("## Prediction Examples")
                gr.Examples(
                    examples = examples_pred,
                    inputs = predict_img,
                    cache_examples = False,
                )
    with gr.Tab("Saved Prompt"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                            #title
                            """
                            # THIS PAGE SHOWS ALL THE SAVED VERSION
                            """
                        )    

demo.launch()
