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
            ['./images/hmbb_1.jpg', './images/hmbb_2.jpg', './images/hmbb_3.jpg'],
            ['./images/rainbow_1.jpg', './images/rainbow_2.jpg', './images/rainbow_3.jpg'],
            ['./images/earth_1.jpg', './images/earth_2.jpg', './images/earth_3.jpg'],
            ['./images/obj_1.jpg', './images/obj_2.jpg', './images/obj_3.jpg'],
            ['./images/ydt_2.jpg', './images/ydt_1.jpg', './images/ydt_3.jpg'],
           ]

examples_sam = [
            ['./images/hmbb_1.jpg', './images/hmbb_2.jpg', './images/hmbb_3.jpg'],
            ['./images/street_1.jpg', './images/street_2.jpg', './images/street_3.jpg'],
            ['./images/tom_1.jpg', './images/tom_2.jpg', './images/tom_3.jpg'],
            ['./images/earth_1.jpg', './images/earth_2.jpg', './images/earth_3.jpg'],
            ['./images/ydt_2.jpg', './images/ydt_1.jpg', './images/ydt_3.jpg'],
           ]

examples_video = [
            ['./videos/horse-running.jpg', './videos/horse-running.mp4'],
            ['./videos/a_man_is_surfing_3_30.jpg', './videos/a_man_is_surfing_3_30.mp4'],
    ['./videos/a_car_is_moving_on_the_road_40.jpg', './videos/a_car_is_moving_on_the_road_40.mp4'],
['./videos/jeep-moving.jpg', './videos/jeep-moving.mp4'],
['./videos/child-riding_lego.jpg', './videos/child-riding_lego.mp4'],
['./videos/a_man_in_parkour_100.jpg', './videos/a_man_in_parkour_100.mp4'],
]

'''
def demo_function(prompt1,prompt2,prompt3,prompt4,prompt5,prompt6,prompt7,prompt8,prompt9,prompt10, img):
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
    
    return img
'''
def demo_function(prompts, img):
    prompt_image_list = []
    prompt_tgt_list = []
    
    for prompt in prompts:
        if prompt is not None:
            prompt_image_list.append(resizeImg(prompt["image"]))
            prompt_tgt_list.append(resizeImg(prompt["mask"]))
    
    # Process the prompts and final image here and return the output image
    return img



demo_1 = gr.Interface(fn = demo_function,
                    inputs=[gr.ImageMask(brush_radius=8, label="prompt1"), gr.ImageMask(brush_radius=8,label="prompt2"), gr.ImageMask(brush_radius=8, label="prompt3"), gr.ImageMask(brush_radius=8,label="prompt4"), gr.ImageMask(brush_radius=8, label="prompt5"), gr.ImageMask(brush_radius=8,label="prompt6"), gr.ImageMask(brush_radius=8, label="prompt7"), gr.ImageMask(brush_radius=8,label="prompt8"), gr.ImageMask(brush_radius=8, label="prompt9"), gr.ImageMask(brush_radius=8,label="prompt10"),gr.Image(label="img")],
                    outputs=[gr.Image(label="output1").style(height=256, width=256)],
                    # outputs=[gr.Image(label="output3 (输出图1)").style(height=256, width=256), gr.Image(label="output4 (输出图2)").style(height=256, width=256)],
                    examples=examples_sam,
                    #title="SegGPT for Any Segmentation<br>(Painter Inside)",
                    description="<p> \
                    <strong>SAM+SegGPT: One touch for segmentation in all images or videos.</strong> <br>\</p>",
                   cache_examples=False,
                   allow_flagging="never",
                   )

'''

samSegGPT = gr.Interface(fn=inference_mask1_sam,
                   inputs=[gr.ImageMask(brush_radius=4, label="prompt (提示图)"), gr.Image(label="img1 (测试图1)"), gr.Image(label="img2 (测试图2)")],
                    outputs=[gr.Image(label="SAM output (mask)").style(height=256, width=256),gr.Image(label="output1 (输出图1)").style(height=256, width=256), gr.Image(label="output2 (输出图2)").style(height=256, width=256)],
                    # outputs=[gr.Image(label="output3 (输出图1)").style(height=256, width=256), gr.Image(label="output4 (输出图2)").style(height=256, width=256)],
                    examples=examples_sam,
                    #title="SegGPT for Any Segmentation<br>(Painter Inside)",
                    description="<p> \
                    <strong>SAM+SegGPT: One touch for segmentation in all images or videos.</strong> <br>\</p>",
                   cache_examples=False,
                   allow_flagging="never",
                   )
'''
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
    return [gr.ImageMask.update(visible=True)]*x+ [gr.ImageMask.update(visible=False)]*(max_imagebox-x)


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
                prompt_button = gr.Button(value = "Submit Amount of Prompt")  
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
                        masked_image = gr.ImageMask(brush_radius=8, label = "Prompt")
                        imagebox.append(masked_image)
                        
            with gr.Column():
                predict_img = gr.Image(shape=(240, 240),label = "Image")
                predict_button = gr.Button('Predict')
                prediction_result = gr.Image(label = "Prediction Result", height = 256)

            prompt_button.click(variable_outputs, slider, imagebox)
            print("HI")
            print(imagebox)
            
            #predict_button.click(demo_function, inputs= [imagebox, predict_img], outputs=prediction_result)
               
    with gr.Tab("Save"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                            #title
                            """
                            # THIS PAGE SHOWS ALL THE SAVED VERSION
                            """
                        )    

demo.launch()
'''
with gr.Blocks() as demo:
    slider = gr.Slider(1,max_imagebox, value = max_imagebox, step = 1, label = "Amount of Prompts: ")
    imagebox = []
    for i in range(max_imagebox):
        t = gr.ImageMask(brush_radius=8)
        imagebox.append(t)
    demo_interface = gr.Interface(fn=demo_function,
                  inputs=[gr.ImageMask.update(visible=True)]*x+ [gr.ImageMask.update(visible=False)]*(max_imagebox-x)+["image"],
                  outputs=[gr.Image(label="output1").style(height=256, width=256)],
                  layout="vertical",
                  title=title)

    slider.change(fn = variable_outputs, inputs = slider, outputs = demo_interface)

    #io1 = gr.TabbedInterface([demo_1],['demo1'],title = title)
  

'''
'''
demo_interface = gr.Interface(fn=demo_function,
                  inputs=[gr.ImageMask(brush_radius=8, label=f"prompt{i+1}") for i in range(max_imagebox)]+["image"],
                  outputs=[gr.Image(label="output1").style(height=256, width=256)],
                  layout="vertical",
                  title=title)

def update_inputs(slider_value):
    inputs = variable_inputs(slider_value)
    demo_interface.inputs = inputs

slider.update(update_inputs, "value")

demo_interface.launch()



with gr.Interface(fn=demo_function,
                  inputs=[gr.ImageMask(brush_radius=8, label=f"prompt{i+1}") for i in range(max_imagebox)]+["image"],
                  outputs=[gr.Image(label="output1").style(height=256, width=256)],
                  layout="vertical",
                  title=title) as demo_interface:
    
    image_boxes = []
    for _ in range(max_imagebox):
        image_box = gr.ImageMask(brush_radius = 8)
        image_boxes.append(image_box)
    slider.change(variable_outputs, slider, image_boxes)
    demo_interface.launch()
'''



'''
with gr.Blocks() as demo:
    slider = gr.Slider(1,max_imagebox, value = max_imagebox, step = 1, label = "Amount of Prompts: ")
    imagebox = []
    for i in range(max_imagebox):
        t = gr.ImageMask(brush_radius=8)
        imagebox.append(t)
    slider.change(variable_outputs, slider, imagebox)

    io1 = gr.TabbedInterface([demo_1],['demo1'],title = title)
    
demo.launch(enable_queue=True)
'''


'''
with gr.Blocks() as demo:
    slider = gr.Slider(1,max_imagebox, value = max_imagebox, step = 1, label = "Amount of Prompts: ")
    imagebox = []
    for i in range(max_imagebox):
        t = gr.ImageMask(brush_radius=8)
        imagebox.append(t)
    slider.change(variable_outputs, slider, imagebox)

    #interface, label, title 
    
    save_button = gr.Button(value = "Save Prompt")
    save_button.click(save_prompt_handler)
    load_button = gr.Button(value = "Load Prompt")
    load_button.click(load_prompt_handler)
    io1 = gr.TabbedInterface(
        [samSegGPT], 
        ['SAM+SegGPT (一触百通)'], 
        title=title,
    )
'''

#demo.launch(share=True, auth=("baai", "vision"))

#demo.launch(enable_queue=False, server_name="0.0.0.0", server_port=34311)

#in inference_mask1_pil(can send empty prompt), add to 10 prompts
#input also set to 10 prompts, and then according to slider(change visibility)
#input can add int 