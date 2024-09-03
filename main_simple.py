from openai import OpenAI
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import shutil
from collections import Counter
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('CHATGPT_API_KEY'))

# postfixes = ['_orig', '_orig_with_bb', '_crop', '_double_context']
postfixes = ['_orig', '_orig_with_bb', '_crop']

class ResponseFormat(BaseModel):
    object_class_name: str

# def get_object_class_name_(image_path: str) -> str:
    # # Open and encode the image
    # img_with_bb = Image.open(image_path)
    # buffer = BytesIO()
    # img_with_bb.save(buffer, format="JPEG")
    # buffer.seek(0)
    # encoded_image_with_bb = base64.b64encode(buffer.read()).decode("utf-8")

    # # Define the prompt
    # prompt = (
    #     "Is there a real human, or a part of a real human in the red bounding box in the center of the image?\n"
    #     "I'm specifically interested in knowing whether there's a real human, or some part of a real human, INSIDE the actual red bounding box. "
    #     "If there are humans right next to the bounding box, but the box itself does not contain any part of a human, then the answer should be NO.\n"
    #     "Note that the person might be occluded or partially visible, but there still is a person inside the bounding box even if the person is occluded or partially visible. "
    #     "Do not confuse with objects that are outside the red bounding box.\n"
    #     "Note that the person might be riding a bicycle, a motorcycle, or sitting inside a car.\n"
    #     "object_class_name should be 'ped' if there's ANY part of pedestrian inside the bounding box, otherwise 'fa'\n"
    # )

    # # Create the content with the image
    # content = [
    #     prompt,
    #     *map(lambda x: {"image": x}, [encoded_image_with_bb]),
    # ]

    # # Make the API call
    # completion = client.beta.chat.completions.parse(
    #     model="gpt-4o-2024-08-06",
    #     messages=[
    #         {"role": "user", "content": content},
    #     ],
    #     response_format=ResponseFormat,
    # )


    # # Parse the response and return the object_class_name
    # gpt_responses = completion.choices[0].message.parsed
    # return gpt_responses.object_class_name

def get_object_class_name(image_path: str) -> str:
    # img_with_bb = Image.open(image_path1)
    # if img_with_bb.mode == 'RGBA':
    #     img_with_bb = img_with_bb.convert('RGB')

    # buffer = BytesIO()
    # img_with_bb.save(buffer, format="JPEG")
    # buffer.seek(0)
    # encoded_image_with_bb = base64.b64encode(buffer.read()).decode("utf-8")

    def encode_image(im_path):
        with open(im_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    encoded_images = []
    # postfixes_to_use = ['_orig', '_crop']
    postfixes_to_use = ['_orig', '_orig_with_bb', '_crop']
    # postfixes_to_use = postfixes
    for p in postfixes_to_use:
        image_path_p = image_path + p + '.png'
        encoded_images.append(encode_image(image_path_p))

    # Define the prompt
    # prompt = (
    #     "There are 3 images attached. The first image is the original image, the second image is the original image with a red bounding box in the center\n"
    #     "The third image is the crop of the red bounding box in the original image.\n"

    #     "Is there a real human, or a part of a real human in the red bounding box in the center of the image?\n"
    #     "I'm specifically interested in knowing whether there's a real human, or some part of a real human, INSIDE the actual red bounding box. "
    #     "If there are humans right next to the bounding box, but the box itself does not contain any part of a human, then the answer should be NO.\n"
    #     "Note that the person might be occluded or partially visible, but there still is a person inside the bounding box even if the person is occluded or partially visible. "
    #     "Do not confuse with objects that are outside the red bounding box.\n"
    #     "Note that the person might be riding a bicycle, a motorcycle, or sitting inside a car.\n"
    #     "answer should be 'ped' if there's ANY part of pedestrian inside the bounding box, otherwise 'fa'. only answer one of these specific phrases. \n"
        
        # "object_class_name should be 'ped' if there's ANY part of pedestrian inside the bounding box, otherwise 'fa'\n"
        # "I also attached a separate image of the crop itself which corresponds to the red bounding box in the main image.\n"
    # )

    # prompt1 = (
    #     "Describe in detail what do you see inside the red bounding box in the center of the image.\n"
    #     # "if there's ANY part of pedestrian inside the bounding box, answer 'ped', otherwise 'fa'.\n"
    # )

    prompt1 = (
    "There are 3 images attached. The first image is the original image, the second image is the original image with a red bounding box in the center\n"
    "The third image is the crop of the red bounding box in the original image.\n"

    "Please carefully describe in detail the contents inside the red bounding box in the first image.\n"
    "Also describe the immediate surroundings around the red bounding box in the first image.\n"
    "Do not make any assumptions about what might be present; just describe exactly what you see.\n"
    "Be as specific as possible, mentioning any visible objects, shapes, textures, colors, and positions.\n"
    )

#     prompt = (
#     "There are 3 images attached. The first image is the original image, the second image is the original image with a red bounding box in the center.\n"
#     "The third image is the crop of the red bounding box in the original image.\n\n"

#     "This task has two steps. You must perform each step **independently**, without considering the information or question in Step 2 while processing Step 1. Treat each step as a separate, independent task.\n\n"

#     "For step 1, don't actually write the answer, but only perform the needs analysis to theoretically answer. For step 2, provide the actual answer.\n\n"
#     "The *only* output for this prompt should be the answer to Step 2.\n\n"

#     "**Step 1: Description**\n"
#     "Carefully describe in detail the contents inside the red bounding box in the second image.\n"
#     "Also, describe the immediate surroundings around the red bounding box in the second image.\n"
#     "Do not make any assumptions about what might be present; just describe exactly what you see.\n"
#     "Be as specific as possible, mentioning any visible objects, shapes, textures, colors, and positions.\n"
#     "Do not consider Step 2 while performing this step.\n\n"

#     "**Step 2: Decision**\n"
#     "Based on your detailed description from Step 1 and considering the third image (crop of the red bounding box), determine if there is any part of a real human (pedestrian) inside the red bounding box in the second image.\n\n"
    
#     "While making your decision in this step, follow these criteria:\n"
#     "- A human might be partially visible, occluded, or sitting inside a vehicle. If there is any doubt or if you are unsure whether a human is present inside the red bounding box, you should answer fa.\n"
#     "- Be cautious not to confuse non-human objects, such as bicycles, motorcycles, poles, signs, bags, shadows, or reflections, for parts of a human.\n"
#     "- If there are humans or parts of humans adjacent to but not inside the red bounding box, the answer should be fa.\n"
#     "- Note that the person might be riding a bicycle, a motorcycle, or sitting inside a car.\n"
#     "Do not use the question in this step to bias or influence your answer in Step 1.\n\n"

#     "Your answer should only be one of these specific phrases: ped if there is ANY part of a pedestrian clearly inside the bounding box, or fa if there is NO part of a pedestrian inside the bounding box or if there is any uncertainty."
# )


    ####### one prompt
    # user_message_content = [{"type": "text", "text": prompt}] + [
    #     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}", "detail": "high"}}
    #     for encoded_image in encoded_images
    # ]

    # # Create the completion request
    # completion = client.chat.completions.create(
    #     model="gpt-4o",
    #     # model="gpt-4o-2024-08-06",
    #     messages=[
    #         # {"role": "system", "content": "You are a computer vision expert that knows how to analyze images and notice little details."},
    #         {"role": "user", "content": user_message_content}
    #     ],
    #     # temperature=0.5,
    # )

    # decision_response = completion.choices[0].message.content

    ################ TWO PROMPTS METHOD ##############
    # Prepare the user message for the first prompt
    user_message_content1 = [{"type": "text", "text": prompt1}] + [
        # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}", "detail": "high"}}
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        for encoded_image in encoded_images
    ]

    # Create the completion request for the first prompt
    completion1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": user_message_content1}
        ],
        # temperature=0.2,
    )

    # Extract the detailed description from the first response
    description_response = completion1.choices[0].message.content
    # print("Description from Prompt 1:", description_response)

    # Prepare the user message for the second prompt
    prompt2 = (
    "You are provided with 3 images and a detailed description of the contents inside and around a red bounding box in the first image.\n"
    "The first image is the original image, the second image is the original image with a red bounding box in the center\n"
    "The third image is the crop of the red bounding box in the original image.\n"
    # "You are provided with 2 images and a detailed description of the contents inside and around a red bounding box in the first image.\n"
    # "The first image is the original image, the second image is the crop of the red bounding box in the original image.\n\n"
    "Below is a description of what is inside and around the red bounding box in the first image:\n\n"

    f"{description_response}\n\n"

    "Based on the images and the description provided, is there a real human, or a part of a real human in the red bounding box in the center of the image?\n"
    "I'm specifically interested in knowing whether there's a real human, or some part of a real human, INSIDE the actual red bounding box. "
    "If there are humans right next to the bounding box, but the box itself does not contain any part of a human, then the answer should be NO.\n"
    "Note that the person might be occluded or partially visible, but there still is a person inside the bounding box even if the person is occluded or partially visible. "
    "Do not confuse with objects that are outside the red bounding box.\n"
    "Note that the person might be riding a bicycle, a motorcycle, or sitting inside a car.\n"
    "Be cautious not to confuse non-human objects, such as bicycles, motorcycles, poles, signs, bags, shadows, or reflections, for parts of a human.\n"
    "answer should be ped if there's ANY part of pedestrian inside the bounding box, otherwise fa. only answer one of these specific phrases: ped/fa\n"
    )

    user_message_content2 = [{"type": "text", "text": prompt2}] + [
        # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}", "detail": "high"}}
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        for encoded_image in encoded_images
    ]

    # Create the completion request for the second prompt
    completion2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": user_message_content2}  
        ],
        # temperature=0.2,
    )

    # Extract the decision from the second response
    decision_response = completion2.choices[0].message.content
    # print("Decision from Prompt 2:", decision_response)
    ################################################

    return decision_response  # Returning the raw response

def process_images_in_folder(main_folder: str):
    results = []

    for subfolder in os.listdir(main_folder):
        if not subfolder.startswith('crops_with_bb'):
            continue

        # if 'crops_with_bb_false' not in subfolder:
        #     continue

        print(f'Processing subfolder: {subfolder}')
        subfolder_path = os.path.join(main_folder, subfolder)

        # Extract initial prediction and manual GT from the folder name
        folder_parts = subfolder.split('_')
        initial_pred = folder_parts[3].lower()  # Assuming INITIAL_METHOD_PRED is the 3rd part
        initial_pred = 'ped' if initial_pred == 'true' else 'fa'

        manual_gt = folder_parts[-1].lower()    # Assuming ManualMANUAL_GT is the last part
        manual_gt = 'ped' if manual_gt == 'manualtrue' else 'fa'

        ## Get all images in the subfolder
        images = os.listdir(subfolder_path)
        images = [x for x in images if x.endswith(('.png', '.jpg', '.jpeg'))]
        # strip all postfixes
        for i, image_file in enumerate(images):
            for p in postfixes:
                if p in image_file:
                    images[i] = image_file.replace(p, '').split('.')[0]
        images = list(set(images))
        ##################################    

        # images = images[:]
        for i, image_file in enumerate(tqdm(images, desc="Processing images", unit="image")):
            image_path = os.path.join(subfolder_path, image_file)
            # if 'ab_car_munich_urban_day_010_001320_det_7861.png' not in image_path:
                # continue

            object_class_name = get_object_class_name(image_path)
            results.append({
                "image_path": image_path + "_orig_with_bb.png",
                "initial_pred": initial_pred,
                "manual_gt": manual_gt,
                "gpt_pred": object_class_name,
            })

    return results


def calculate_metrics(y_true, y_pred, label_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='ped', average='binary')
    recall = recall_score(y_true, y_pred, pos_label='ped', average='binary')
    f1 = f1_score(y_true, y_pred, pos_label='ped', average='binary')
    print(f"\nMetrics for {label_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1


def create_error_directory_structure(main_folder: str, error_folder: str):
    if os.path.exists(error_folder):
        shutil.rmtree(error_folder)
    os.makedirs(error_folder)
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            new_subfolder_path = os.path.join(error_folder, subfolder)
            os.makedirs(new_subfolder_path)


def copy_errors_to_directory(results, error_folder):
    for result in results:
        if result['gpt_pred'] != result['manual_gt']:
            subfolder_name = os.path.basename(os.path.dirname(result['image_path']))
            new_image_path = os.path.join(error_folder, subfolder_name, os.path.basename(result['image_path']))
            shutil.copy(result['image_path'], new_image_path)
            # print(f"Copied {result['image_path']} to {new_image_path}")


if __name__ == "__main__":
    main_folder = "/Users/tomercohen/Downloads/crop2vec_chatgpt/Manual_Tagged_3vars_5x"
    error_folder = "/Users/tomercohen/Downloads/crop2vec_chatgpt/Manual_Tagged_ChatGPT_errors"
    
    # Create an empty clone of the input directories
    create_error_directory_structure(main_folder, error_folder)

    results = process_images_in_folder(main_folder)
    print(f'unique gpt preds: {Counter([result["gpt_pred"] for result in results])}')

    # Collect labels and predictions for comparison
    initial_preds = [result['initial_pred'] for result in results]
    manual_gts = [result['manual_gt'] for result in results]
    gpt_preds = [result['gpt_pred'] for result in results]

    # Calculate metrics for Initial Method vs. Manual GT
    calculate_metrics(manual_gts, initial_preds, "Initial Method")

    # Calculate metrics for GPT-4 Prediction vs. Manual GT
    calculate_metrics(manual_gts, gpt_preds, "GPT-4 Prediction")

    # Copy images where GPT-4 predictions were wrong
    copy_errors_to_directory(results, error_folder)
