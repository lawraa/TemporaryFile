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
import h5py
from datetime import datetime
import re
from SegGPT_inference.seggpt_inference import prepare_model
from SegGPT_inference.seggpt_engine import inference_image_pil

#Date and Time

'''
# datetime object containing current date and time
now = datetime.now()
 
print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

#now = 2023-07-18 14:16:55.857082
#date and time = 18/07/2023 14:16:55
'''

sys.path.append('.')



#Resize Image Function 
#    - Used for prompt sending into model
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

# Examples of Prompts and Prediction Function
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

#Load Model (When use ---> Un-comment every line)

#device = "cuda"
#model = prepare_model("SegGPT_inference/seggpt_vit_large.pth", "seggpt_vit_large_patch16_input896x448", "instance").to(device)
#print('Model loaded.')  

# Dummy model function
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
        
    #Un-Comment when use GPU and load model   
    output = inference_image_pil(model, device, resizeImg(img), prompt_image_list, prompt_tgt_list)
    
    return img

#Better way of sending in prompts(Error - list error)
def demo_function(prompts, img):
    prompt_image_list = []
    prompt_tgt_list = []
    
    for prompt in prompts:
        if prompt is not None:
            prompt_image_list.append(resizeImg(prompt["image"]))
            prompt_tgt_list.append(resizeImg(prompt["mask"]))
    
    # Process the prompts and final image here and return the output image
    return img

#Other Functions

#Function that evaluates the performance
def evaluate_function(img):
    #after evaluation
    x=0.488
    return x

#Help with slider prompt
def variable_outputs(x):
    x = int(x)
    return [gr.ImageMask.update(visible=True)]*x+ [gr.ImageMask.update(value= None, visible=False)]*(max_imagebox-x)

#get user name and project

def get_cookie_value(cookie_string, cookie_name):
    pattern = cookie_name + r'=(.*?)(?:;|$)'
    match = re.search(pattern, cookie_string)
    if match:
        return match.group(1)
    else:
        return ''

def get_user_name(request: gr.Request): #returns the user name
    cookie_string = request.headers['cookie']
    user_name = get_cookie_value(cookie_string, 'user')
    user_name = base64.b64decode(user_name).decode('utf-8')
    return user_name, user_name

def get_projects(user_name): #returns the dropdown of list of projects
    print('user_name in get_projects', user_name)
    url = "http://172.18.212.157:31589/prjcfg/ProjectList_Owner_and_WhiteUser"
    params = {
        "request": user_name,
        "filter_solution": "true"
    }
    headers = {
        "Accept": "application/json"
    }

    response = requests.get(url, params=params, headers=headers)
    project_list = response.json().get('PRJ_LIST', []) #this list contains all the projects
    #project_names_token_pair = [p.get('NAME', '') + ' - ' + p.get('TOKEN', '') for p in project_list] #gives a list of all projects one by one
    project_names_token_pair = [p.get('NAME', '') for p in project_list]
    return gr.Dropdown.update(choices=project_names_token_pair), project_list

def show_project_token(project_name_token):
    token = project_name_token.split(' - ')[1]
    return token


#Package everything into a HDF file

def package_images(prompt1, prompt2, prompt3, prompt4, prompt5,prompt6,prompt7,prompt8,prompt9,prompt10, output_file,
                    predict_img,prediction_result, answer_mask, performance):
    # Create a new HDF5 file
    with h5py.File(output_file, 'w') as hf:
        # Create groups and store the image data
        prompts = hf.create_group("prompts")
        test_set = hf.create_group("test_set")
        img = prompts.create_group("img")
        mask = prompts.create_group("mask")
        pred = test_set.create_group("pred")
        pred_ans = test_set.create_group("pred_ans")
        pred_img = pred.create_group("pred_img")
        pred_result_img = pred.create_group("pred_result_img")
        #pred_result_mask = pred.create_group("pred_result_mask")
        print("start")
        if prompt1:
            img.create_dataset('image_1', data=prompt1['image'])
            mask.create_dataset('mask_1', data=prompt1['mask'])
            print("file1")
        if prompt2:    
            img.create_dataset('image_2', data=prompt2['image'])
            mask.create_dataset('mask_2', data=prompt2['mask'])
            print("file2")
        if prompt3:
            img.create_dataset('image_3', data=prompt3['image'])
            mask.create_dataset('mask_3', data=prompt3['mask'])
            print("file3")
        if prompt4:
            img.create_dataset('image_4', data=prompt4['image'])
            mask.create_dataset('mask_4', data=prompt4['mask'])
            print("file4")
        if prompt5:
            img.create_dataset('image_5', data=prompt5['image'])
            mask.create_dataset('mask_5', data=prompt5['mask'])
            print("file5")
        if prompt6:
            img.create_dataset('image_6', data=prompt6['image'])
            mask.create_dataset('mask_6', data=prompt6['mask'])
            print("file6")
        if prompt7:
            img.create_dataset('image_7', data=prompt7['image'])
            mask.create_dataset('mask_7', data=prompt7['mask'])
            print("file7")
        if prompt8:
            img.create_dataset('image_8', data=prompt8['image'])
            mask.create_dataset('mask_8', data=prompt8['mask'])
            print("file8")
        if prompt9:
            img.create_dataset('image_9', data=prompt9['image'])
            mask.create_dataset('mask_9', data=prompt9['mask'])
            print("file9")
        if prompt10:
            img.create_dataset('image_10', data=prompt10['image'])
            mask.create_dataset('mask_10', data=prompt10['mask'])
            print("file10")

        

        if predict_img.any():
            print("hi")
            pred_img.create_dataset('pred_img_1', data = predict_img)
            if performance:
                print("hi")
                pred_img.attrs['performance'] = performance
        if prediction_result.any():
            print("hi")
            #pred_result_img.create_dataset('prediction_result_img', data= prediction_result['image'])
            #pred_result_mask.create_dataset('prediction_result_mask', data= prediction_result['mask'])
            pred_result_img.create_dataset('prediction_result_img', data= prediction_result)
        if answer_mask.any():
            print("hi")
            pred_ans.create_dataset('pred_ans', data = answer_mask)


#Helper Function for packaging and downloading
def download_h5_file(prompt1, prompt2, prompt3, prompt4, prompt5,prompt6,prompt7,prompt8,prompt9,prompt10,predict_img,
                        prediction_result, answer_mask, performance, prompt_name):
    # Package the images into an HDF5 file
    output_file = prompt_name + '.h5'
    package_images(prompt1, prompt2, prompt3, prompt4, prompt5,prompt6,prompt7,prompt8,prompt9,prompt10, output_file, predict_img, prediction_result, answer_mask, performance)
    return output_file


demo_title = "<div align='center'> <h1 style='font-size:60px;'> Demo </h1>\
<h2>This is our demo blah blah blah</h2> \
<h2>jfsdgnkjsdnfksdfnvksdkjnvlnewjgernjknver</h2> \
</div> \
"

project_info_list = []

#Determine Maximum Amount of Prompts
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

#gradio 
css = """
#warning {background-color: #FFCCCB} 
.login textarea {position: relative/absolute; top: #px; right: #px;}

"""

with gr.Blocks() as demo:
    #css=".gradio-container {background-color: #D3D3D3}"
    user_name = gr.State('None')
    project_options = gr.State([])
    project_info_list = gr.State([])
    
    with gr.Row():
        with gr.Column():
            gr.Text(visible=False)
        with gr.Column():
            gr.Text(visible=False)
        with gr.Column():
            user_name_display = gr.Text(label="user_name",elem_classes="login")
            project_dropdown = gr.Dropdown(project_options.value, label='Project Names', interactive=True)
    
    demo.load(
        fn=get_user_name,
        inputs=[],
        outputs=[user_name, user_name_display]
    )
    demo.load(
        fn=get_projects,
        inputs=[user_name],
        outputs=[project_dropdown, project_info_list]
    )
    print(user_name)
    print(project_info_list)
    gr.Markdown(
        demo_title
    )
    with gr.Tab("Prompting and Predicting"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """ # Upload and Predict """
                )
                gr.Markdown(
                    '''
                    ## Slide to Adjust the Amount of Prompts you want to Upload
                    ## Upload your Prompts
                    '''
                )
        with gr.Row():
            with gr.Column():
                slider = gr.Slider(1,max_imagebox, step = 1, value = 10, label = "Amount of Prompts: ")
                prompt_button = gr.Button(value = "Submit Amount of Prompt")                
        with gr.Row():    
            imagebox = []
            for i in range(max_imagebox):
                masked_image = gr.ImageMask(brush_radius=8, label = "Prompt", min_width = 350)
                imagebox.append(masked_image)     
        with gr.Row():
            gr.Markdown(
                '''
                <br>
                <br>
                <br>
                <h1> Upload the Image to Predict </h1>
                '''
            )        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    predict_img = gr.Image(shape=(240, 240),label = "Image")
                with gr.Row():
                    predict_button = gr.Button('Predict')
                    #clear_prompt_button = gr.ClearButton(imagebox, value = 'Clear All Prompt')
                    #clear_image_button = gr.ClearButton(predict_img, value = 'Clear Predict Image')
            with gr.Column():
                with gr.Row():
                    prediction_result = gr.Image(label = "Prediction Result")
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        answer_mask = gr.Image(label = "Labeled Answer")
                        evaluate_button = gr.Button('Evaluate Performance')
                        performance = gr.Textbox(label = "Performance(Dice Coefficient)")   
        with gr.Row():
            gr.Markdown(
                '''
                <br>
                <br>
                <br>
                <h1> Load or Save Prompts </h1>
                '''
            )     
        with gr.Row():
            prompt_name = gr.Textbox(label = "Prompt_name")
            save_button = gr.Button('Save Prompts')
            

        with gr.Row():
            with gr.Column():
                output = gr.File(Label = "Download HDF5")

            evaluate_button.click(evaluate_function, answer_mask, performance)

            

#SLIDE THEN CLICK TO CHANGE
            prompt_button.click(variable_outputs, slider, imagebox)

#SLIDE AND CHANGE IMMEDIATELY
            #slider.change(variable_outputs, slider, imagebox)

            
            print("HI")
            print(imagebox)
            #predict_button.click(demo_function, inputs= [imagebox, predict_img], outputs=prediction_result)
            predict_button.click(demo_function2, inputs= [imagebox[0], imagebox[1],imagebox[2],imagebox[3],imagebox[4],imagebox[5],imagebox[6],imagebox[7],imagebox[8],imagebox[9], predict_img], outputs=[prediction_result])
            #load_button.click()

            save_button.click(download_h5_file, inputs= [imagebox[0], imagebox[1],imagebox[2],imagebox[3],imagebox[4],imagebox[5],imagebox[6],imagebox[7],imagebox[8],imagebox[9],
                                             predict_img,prediction_result, answer_mask, performance,prompt_name], outputs = output)

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

demo.launch(enable_queue=True, server_name="0.0.0.0",server_port=6006)
