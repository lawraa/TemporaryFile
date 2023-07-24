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
from cambrian import Dataset
from SegGPT_inference.seggpt_inference import prepare_model
from SegGPT_inference.seggpt_engine import inference_image_pil


#Date and Time
#Lawrance
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


# +
#Resize Image Function 
#    - Used for prompt sending into model
def resizeImg_(img):
    res, hres = 448, 448
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    temp = io.BytesIO()
    img.save(temp, format="WEBP")
    return base64.b64encode(temp.getvalue()).decode('ascii')

def resizeImg(img):
    res, hres = 448, 448
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    return img

def resizeImgIo(img):
    res, hres = 448, 448
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    temp = io.BytesIO()
    img.save(temp, format="WEBP")
    return io.BytesIO(temp.getvalue())


# -

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

# +
#Load Model (When use ---> Un-comment every line)

#device = "cuda"
#model = prepare_model("SegGPT_inference/seggpt_vit_large.pth", "seggpt_vit_large_patch16_input896x448", "instance").to(device)
#print('Model loaded.')  
# -



# +
# Dummy model function
def demo_function2(prompt1,prompt2,prompt3,prompt4,prompt5,prompt6,prompt7,prompt8,prompt9,prompt10,
                 img):
    prompt_image_list = []
    prompt_tgt_list = []
    variable_type = type(prompt1["image"])
    variable_type2 = type(prompt1["mask"])
    # Print the type
    print(variable_type)
    print(variable_type2)
    if prompt1:
        print("start")

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
    #output = inference_image_pil(model, device, resizeImg(img), prompt_image_list, prompt_tgt_list)
    output = img
    return output

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


# +
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


# +
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

def get_projects(user_name):
    print('user_name in get_projects', user_name)
    url = "http://cambrian-project-mgr-api-service.cambrian-platform/prjcfg/ProjectList_Owner_and_WhiteUser"
    params = {
        "request": user_name,
        "filter_solution": "true"
    }
    headers = {
        "Accept": "application/json"
    }

    response = requests.get(url, params=params, headers=headers)
    project_list = response.json().get('PRJ_LIST', [])
    project_names_token_pair = [p.get('NAME', '') + ' - ' + p.get('TOKEN', '') for p in project_list]
    project_names = [p.get('NAME', '') for p in project_list]
    return gr.Dropdown.update(choices=project_names_token_pair), project_list

def show_all_dataset(p_token):
    #LAWRANCE
    print("show all dataset function")
    dataset = load_data(p_token)
    name_list = []
    for key, value in dataset.items():
        name = value.get('name')
        if name:
            dataset_id = key
            name_with_id = f"{name} - {dataset_id}"
            #print(name_with_id)
            name_list.append(name_with_id)    

    return gr.Dropdown.update(choices=name_list)
    
    
    #LAWRANCE
def get_data(d_token,p_token):
    ds = Dataset()
    filepath = '../file.h5'
    ret = ds.get(dataset_token= d_token , token = p_token, version=None, filename=filepath)
    ret

def show_project_token(project_name_token):
    token = project_name_token.split(' - ')[1]
    return token

def show_dataset_token(dataset_name_token):
    token = dataset_name_token.split(' - ')[1]
    return token

def show_project_name(project_name_token):
    token = project_name_token.split(' - ')[0]
    return token

def get_path(output_file): #output_file is gr.File variable
    return str(output_file.name)  # Use .name to get the file path from the output_file object


project_token = os.environ['PROJECT_TOKEN']

print(f'Project token: {project_token}')

# save prompt

def save_data(user_name, file_path, project_token, file_name,prompt_num,predict_img_num,performance_num,prompt_desc):
    print("user name: " + user_name)
    print("file path: " + file_path)
    print("project_token: " + project_token)
    print("file_name: " + file_name)

    ds = Dataset()
    ret = ds.add(
                filename=file_path, #your f5 filepath in local
                token=project_token,
                name=file_name,
                visible="private",
                meta={
                    "general":{
                        "image_count": {
                            "train": prompt_num,
                            "validate": 0,
                            "test": predict_img_num
                        },
                        "labeled_image_count": prompt_num,
                        "unlabeled_image_count": 0,
                        "sample_count": { #same as image_count
                            "train": prompt_num,
                            "validate": 0,
                            "test": predict_img_num
                        },
                        "labeled_sample_count": prompt_num, #same as labeled_image_count
                        "unlabeled_sample_count": 0, #same as unlabeled_image_count
                        "description": prompt_desc,
                        "project_token": project_token,
                        "dataset_type": [
                            "image"
                        ]
                    },
                    "image": {
                        "color_type": "Binary",
                        "image_dataset_format": "Binary",
                        "image_resolution": {
                            "448x448": prompt_num #your images size and count
                        }
                    },
                    "label": {
                        "label_type": [
                            "classification",
                            "seggpt" # custome label
                            "gradio_seggpt"
                        ]
                    },
                    "misc": {
                        "creator": user_name, 
                        "job_id": 0,
                        "job_name": "",
                        "resize": "none"
                    }
                    
                    
                }
    )

def load_data(project_token):
    ds = Dataset()
    ret = ds.get_private_dataset(project_token)
    return ret




def load_and_display(project_token):
    print("This is the load and display table function")
    print("Project token passed in is " + project_token)
    dataset = load_data(project_token)
    return create_table(dataset)

def create_table(dataset):
    print("This is the create table function")
    data_list = []
    for key, value in dataset.items():
        row = {'dateset_token': key}
        row.update(value)
        if 'versions' in value and len(value['versions']) > 0:
            version = value['versions'][0]
            metas = version.get('meta', {})
            row['Created Time'] = version.get('createdTime',"None")
            row['Name'] = value.get('name',"None")
            try:
                row['Prompt Image'] = metas['general']['image_count'].get('train')
            except:
                row['Prompt Image'] = 0
            try:
                row['Testing Image'] = metas['general']['image_count'].get('test')
            except:
                row['Testing Image'] = 0
            try:
                row['Labels'] = ', '.join(metas['label']['label_type']) if 'label_type' in metas['label'] else ''
            except:
                row['Labels'] = ''
            try:
                row['Description'] = metas['general'].get('description', '')   
            except:
                row['Description'] = ''

        data_list.append(row)

    df = pd.DataFrame(data_list)

    # Select the desired columns in the specified order
    desired_columns = ['Created Time', 'Name', 'Prompt Image', 'Testing Image', 'Labels', 'Description']
    df = df[desired_columns]

    #df = df[df['label_type'].str.contains('seggpt', case=False, na=False)]

    return df


# +
#Package everything into a HDF file

def package_images(prompt1, prompt2, prompt3, prompt4, prompt5,prompt6,prompt7,prompt8,prompt9,prompt10, output_file,
                    predict_img,prediction_result, answer_mask, performance, description):
    prompt_num = 0
    predict_img_num = 0 
    performance_num = 0
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
        if prompt1:
            img.create_dataset('image_1', data=prompt1['image'])
            mask.create_dataset('mask_1', data=prompt1['mask'])
            print("file1")
            prompt_num+=1
        if prompt2:    
            img.create_dataset('image_2', data=prompt2['image'])
            mask.create_dataset('mask_2', data=prompt2['mask'])
            print("file2")
            prompt_num+=1
        if prompt3:
            img.create_dataset('image_3', data=prompt3['image'])
            mask.create_dataset('mask_3', data=prompt3['mask'])
            print("file3")
            prompt_num+=1
        if prompt4:
            img.create_dataset('image_4', data=prompt4['image'])
            mask.create_dataset('mask_4', data=prompt4['mask'])
            print("file4")
            prompt_num+=1
        if prompt5:
            img.create_dataset('image_5', data=prompt5['image'])
            mask.create_dataset('mask_5', data=prompt5['mask'])
            print("file5")
            prompt_num+=1
        if prompt6:
            img.create_dataset('image_6', data=prompt6['image'])
            mask.create_dataset('mask_6', data=prompt6['mask'])
            print("file6")
            prompt_num+=1
        if prompt7:
            img.create_dataset('image_7', data=prompt7['image'])
            mask.create_dataset('mask_7', data=prompt7['mask'])
            print("file7")
            prompt_num+=1
        if prompt8:
            img.create_dataset('image_8', data=prompt8['image'])
            mask.create_dataset('mask_8', data=prompt8['mask'])
            print("file8")
        if prompt9:
            img.create_dataset('image_9', data=prompt9['image'])
            mask.create_dataset('mask_9', data=prompt9['mask'])
            print("file9")
            prompt_num+=1
        if prompt10:
            img.create_dataset('image_10', data=prompt10['image'])
            mask.create_dataset('mask_10', data=prompt10['mask'])
            print("file10")
            prompt_num+=1
        if predict_img.any():
            pred_img.create_dataset('pred_img_1', data = predict_img)
            predict_img_num +=1
            if performance:
                pred_img.attrs['performance'] = performance
                performance_num += 1
                print("There is a performance")
            if description:
                pred_img.attrs['description'] = description
                print("There is a description")
        if prediction_result.any():
            print("hi")
            #pred_result_img.create_dataset('prediction_result_img', data= prediction_result['image'])
            #pred_result_mask.create_dataset('prediction_result_mask', data= prediction_result['mask'])
            pred_result_img.create_dataset('prediction_result_img', data= prediction_result)
        if answer_mask.any():
            print("hi")
            pred_ans.create_dataset('pred_ans', data = answer_mask)
    return prompt_num, predict_img_num, performance_num


#Helper Function for packaging and downloading
def download_h5_file(prompt1, prompt2, prompt3, prompt4, prompt5,prompt6,prompt7,prompt8,prompt9,prompt10,predict_img,
                        prediction_result, answer_mask, performance, description, prompt_name):
    # Package the images into an HDF5 file
    output_file = prompt_name + '.h5'
    prompt_num, predict_img_num, performance_num = package_images(prompt1, prompt2, prompt3, prompt4, prompt5,prompt6,prompt7,prompt8,prompt9,prompt10, output_file, predict_img, prediction_result, answer_mask, performance,description)
    return output_file, prompt_num, predict_img_num, performance_num

def read_h5_file():
    file_path = "../file.h5"
    with h5py.File(file_path,'r') as hf:
        data_dict = {}
        def extract_data(name, obj):
            if isinstance(obj, h5py.Dataset):
                    data_dict[name] = [obj[()]]

        hdf_file.visititems(extract_data)
        df = pd.DataFrame(data_dict)


def display_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    


# +
demo_title = "<div style = 'text-align: center;margin: 50px auto;padding: 20px;max-width: 1000px;background-color: #ffffff;box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);border-radius: 10px;color: #333333;'> \
<h1 style = 'font-size: 60px;font-weight: bold;color: #003366;text-shadow: 2px 2px 4px #cccccc;'>Inference-Based Defect Detection</h1>\
<div style = 'font-size: 24px;color: #333333;line-height: 1.6;max-width: 800px;margin: 0 auto;'> By implementing a defect detection AI model on the production line, we aim to streamline operations and improve quality control. Our goals include reducing missing and false reports, cutting down on human resources, and simplifying the overall working process.</div> \
<div style='font-size: 18px;color: #666666;line-height: 1.6;max-width: 800px;margin: 30px auto;'>UI allows users to upload images, labeled or unlabeled, perform defect detection, and report performance given the ground truth.</div> \
</div>\
" 
upload_prompt_title = "<div style = 'text-align: center;margin: 35px auto;padding: 20px;max-width: 800px;background-color: #ffffff;box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);border-radius: 10px;color: #333333;'> \
<h1 style = 'font-size: 35px;font-weight: bold;margin-bottom: 16px;color: #003366;'>Upload Prompt</h1>\
<div style = 'text-align: left;font-size: 20px;line-height: 1.6;margin-bottom: 16px;'> 1. Slide to choose the amount of prompt you want to upload </div> \
<div style='text-align: left;font-size: 20px;line-height: 1.6;margin-bottom: 20px;'>2. Label your prompts</div> \
</div>\
"

upload_predict_image_title = "<div style = 'text-align: center;margin: 35px auto;padding: 20px;max-width: 800px;background-color: #ffffff;box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);border-radius: 10px;color: #333333;'> \
<h1 style = 'font-size: 35px;font-weight: bold;margin-bottom: 16px;color: #003366;'>Image Prediction and Evaluation</h1>\
<div style = 'text-align: left;font-size: 20px;line-height: 1.6;margin-bottom: 16px;'> 1. Upload your image for prediction </div> \
<div style='text-align: left;font-size: 20px;line-height: 1.6;margin-bottom: 20px;'>2. Upload the labeled mask and Evaluate Performance</div> \
</div>\
"

save_prompt_title = "<div style = 'text-align: center;margin: 35px auto;padding: 20px;max-width: 800px;background-color: #ffffff;box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);border-radius: 10px;color: #333333;'> \
<h1 style = 'font-size: 35px;font-weight: bold;margin-bottom: 16px;color: #003366;'>Save Prompts</h1>\
<div style = 'text-align: left;font-size: 20px;line-height: 1.6;margin-bottom: 16px;'> 1. Enter your filename and save your prompt </div> \
<div style='text-align: left;font-size: 20px;line-height: 1.6;margin-bottom: 20px;'>2. Check your saved prompt in 'Saved Prompt' tab </div> \
</div>\
"
css = """
.imageItem {
min-width: 450px !important;
max-width: 450px !important;
}

.svelte-yigbas {
width: 100% !important;
height: 100% !important;
object-fit: contain !important;
}

"""

project_info_list = []

#Determine Maximum Amount of Prompts
max_imagebox = 10

saved_prompts = []


# +
with gr.Blocks(css =css) as demo:
    user_name = gr.State('None')
    project_options = gr.State([])
    project_info_list = gr.State([])
    project_token = gr.State('')
    
    with gr.Row():
        with gr.Column(scale =7):
            gr.Markdown(demo_title)
        with gr.Column(scale =1):
            user_name_display = gr.Text(label="user_name")
            project_dropdown = gr.Dropdown(project_options.value, label='Project Names', interactive=True)
            project_token_display = gr.Text(label="project_token", visible = False)
            dataset_token_display = gr.Text(label="dataset_token", visible = False)
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

    project_dropdown.change(
        fn=show_project_token,
        inputs=[project_dropdown],
        outputs=[project_token_display]
    )
    
    print(user_name)
    print(project_info_list)

    
    gr.Markdown("<br>\
                <br>\
                ")
    with gr.Tab("Prompting and Predicting"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    upload_prompt_title
                )
                
        with gr.Row():
            slider = gr.Slider(1,max_imagebox, step = 1, value = 10, label = "Amount of Prompts: ",scale=5)
            prompt_button = gr.Button(value = "Enter",scale = 1)                
        with gr.Row():    
            imagebox = []
            for i in range(max_imagebox):
                
                masked_image = gr.ImageMask(brush_radius=16, label = "Prompt", elem_classes='imageItem', scale = 1)

                imagebox.append(masked_image)
        with gr.Row():
            gr.Markdown(upload_predict_image_title)        
        with gr.Row():
            '''
            with gr.Column():
                gr.Markdown("# Upload Hdf5 File For Prediction")
                test_file = gr.File(Label = "Upload Hdf5 File for Prediction")
                test_evaluation = gr.Button('Evaluate Performance')
                test_performance = gr.Textbox(label = "Performance(Dice Coefficient)")   
            '''
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("# Upload Photo for Prediction")
                        predict_img = gr.Image(shape=(240, 240),label = "Predict Image")
                        predict_button = gr.Button('Predict')
                        #clear_prompt_button = gr.ClearButton(imagebox, value = 'Clear All Prompt')
                        #clear_image_button = gr.ClearButton(predict_img, value = 'Clear Predict Image')
                    
                        prediction_result = gr.Image(label = "Prediction Result", interactive = False)
                        
            
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("# <br>\
                                    ")
                        answer_mask = gr.Image(label = "Labeled Answer")
                        evaluate_button = gr.Button('Evaluate Performance')
                        performance = gr.Textbox(label = "Performance(Dice Coefficient)",interactive = False)   
        with gr.Row():
            gr.Markdown(save_prompt_title)     
        with gr.Row():
            with gr.Column():
                prompt_name = gr.Textbox(label = "Prompt Name")
                prompt_desc = gr.Textbox(label = "Description")
        with gr.Row():
            with gr.Column():
                gr.Text(visible=False)
            with gr.Column():
                with gr.Row():
                    create_file_button = gr.Button('Create File')
                with gr.Row():
                    save_file_button = gr.Button('Save Prompt')
            with gr.Column():
                gr.Text(visible=False)

                
            

        with gr.Row():
            with gr.Column():
                output_file = gr.File(interactive = False)

            evaluate_button.click(evaluate_function, answer_mask, performance)      

#SLIDE THEN CLICK TO CHANGE
            prompt_button.click(variable_outputs, slider, imagebox)
#SLIDE AND CHANGE IMMEDIATELY
            #slider.change(variable_outputs, slider, imagebox)
    
    
            print(imagebox)
            #predict_button.click(demo_function, inputs= [imagebox, predict_img], outputs=prediction_result)
            predict_button.click(demo_function2, inputs= [imagebox[0], imagebox[1],imagebox[2],imagebox[3],imagebox[4],imagebox[5],imagebox[6],imagebox[7],imagebox[8],imagebox[9], predict_img], outputs=[prediction_result])
            #load_button.click()
            prompt_num = gr.State('')
            predict_img_num = gr.State('')
            performance_num = gr.State('')
            create_file_button.click(download_h5_file, inputs= [imagebox[0], imagebox[1],imagebox[2],imagebox[3],imagebox[4],imagebox[5],imagebox[6],imagebox[7],imagebox[8],imagebox[9],
                                             predict_img,prediction_result, answer_mask, performance,prompt_desc,prompt_name], outputs = [output_file,prompt_num,predict_img_num, performance_num])
            file_path = gr.State('')
            output_file.change(get_path, inputs = [output_file], outputs = file_path)  
            save_file_button.click(save_data, inputs = [user_name, file_path, project_token_display, prompt_name,prompt_num,predict_img_num,performance_num,prompt_desc])
            
        with gr.Row():
            with gr.Column():
                gr.Markdown("<br>\
                <br>\
                ")
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
                            # Add a chart that shows entire chart
                            # Add a dropdown
                            """
                        ) 
                
        with gr.Row():
            table = gr.Dataframe(interactive = False)
            #load_and_display
            
#        table.change(load_and_display, inputs=[project_token_display],outputs=[table])
        project_token_display.change(
            fn=load_and_display,
            inputs=[project_token_display],
            outputs=[table]
        )    
        with gr.Row():
            gr.Markdown(
                            #title
                            """
                            # Load Model Blahblahblah
                            """
                        )
        data_options = gr.State([])
        with gr.Row():
            load_dropdown = gr.Dropdown(data_options.value, label='Project Names', interactive=True)

        project_token_display.change(
            fn=show_all_dataset,
            inputs=[project_token_display],
            outputs=[load_dropdown]
        )  
        
        
        load_dropdown.change(
            fn=show_dataset_token,
            inputs=[load_dropdown],
            outputs=[dataset_token_display]
        )
        
        with gr.Row():
            with gr.Column():
                gr.Text(visible=False)
            with gr.Column():
                load_button = gr.Button("Load Button")
                temp_button = gr.Button("Temp Button")
            with gr.Column():
                gr.Text(visible=False)
        with gr.Row():
            each_data = gr.Dataframe(interactive = False)
            another_data = gr.Dataframe(interactive = False)
            
        load_button.click(get_data, inputs = [dataset_token_display, project_token_display])

        temp_button.click(read_h5_file, outputs = [each_data,another_data])
        
        #gr.load(load_and_display, inputs =project_token_display, outputs = table)        
        #load_button.click(function, inputs, outputs) 
# -

demo.launch(enable_queue=True, server_name="0.0.0.0",server_port=6006)


