from openai import OpenAI
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64
import os
import pickle
import re
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import shutil
from collections import Counter
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('CHATGPT_API_KEY'))

# postfixes = ['_orig', '_orig_with_bb', '_crop']
postfixes = ['_orig', '_context_10', '_context_5', '_context_2', '_crop']

def extract_integer(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return None

def get_object_class_name(image_path: str, model: str) -> str:
    def encode_image(im_path):
        with open(im_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    encoded_images = []
    # postfixes_to_use = ['_orig', '_orig_with_bb', '_crop']
    for p in postfixes:
        image_path_p = image_path + p + '.png'
        encoded_images.append(encode_image(image_path_p))

    prompt1 = (
    # "There are 3 images attached. The first image is the original image, the second image is the original image with a red bounding box in the center\n"
    # "The third image is the crop of the red bounding box in the original image.\n"
    "There are 5 images attached. The first image is the original image, the second image is the original image with a red bounding box in the center\n"
    "The third and forth images are the crops of the red bounding box in the original image with different context sizes.\n"
    "The fifth image is the actual crop of the red bounding box in the original image.\n"

    "Please carefully describe in detail the contents inside the red bounding box in the first image.\n"
    "Consider the context around the bounding box, and the other images to verify your assumption on the bounding box content.\n"
    "Also describe the immediate surroundings around the red bounding box in the first image.\n"
    "Do not make any assumptions about what might be present; just describe exactly what you see.\n"
    "Be as specific as possible, mentioning any visible objects, shapes, textures, colors, and positions.\n"
    )

    ################ TWO PROMPTS METHOD ##############
    # Prepare the user message for the first prompt
    user_message_content1 = [{"type": "text", "text": prompt1}] + [
        # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}", "detail": "high"}}
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        for encoded_image in encoded_images
    ]

    # Create the completion request for the first prompt
    completion1 = client.chat.completions.create(
        model=model,
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
    "You are provided with 5 images and a detailed description of the contents inside and around a red bounding box in the first image.\n"
    # "The first image is the original image, the second image is the original image with a red bounding box in the center\n"
    # "The third image is the crop of the red bounding box in the original image.\n"

    " The first image is the original image, the second image is the original image with a red bounding box in the center\n"
    "The third and forth images are the crops of the red bounding box in the original image with different context sizes.\n"
    "The fifth image is the actual crop of the red bounding box in the original image.\n"

    "Below is a description of what is inside and around the red bounding box in the first image:\n\n"

    f"{description_response}\n\n"

    # ##### best ########
    # "Based on the images and the description provided, is there a real human, or a part of a real human in the red bounding box in the center of the image?\n"
    # "I'm specifically interested in knowing whether there's a real human, or some part of a real human, INSIDE the actual red bounding box. "
    # # "If there are humans right next to the bounding box, but the box itself does not contain any part of a human, then the answer should be NO.\n"
    # # "Note that the person might be occluded or partially visible, but there still is a person inside the bounding box even if the person is occluded or partially visible. "
    # "Do not confuse with objects that are outside the red bounding box.\n"
    # "Note that the person might be riding a bicycle, a motorcycle, or sitting inside a car.\n"
    # "Be cautious not to confuse non-human objects, such as bicycles, motorcycles, poles, signs, bags, shadows, or reflections, for parts of a human.\n"

    # "If you are not sure, please say so explicitly. I want your answer to be 100% correct, so if you have the slightest doubt, tell me you're not sure.\n"

    # "answer should be ped if there's ANY part of human inside the bounding box, otherwise fa. only answer one of these specific phrases: ped/fa/maybe\n"
    # "Also provide a confidence score between 0 and 100, indicating how confident you are in your answer.\n"
    # "The final answer should be in the format: ped CONF, fa CONF, maybe CONF etc.\n"
    # #####################

    "Based on the images and the description provided, is there a real human, or a part of a real human inside the red bounding box?\n"
    "Do not confuse with objects that are outside the red bounding box.\n"
    "If you are not sure, please say so explicitly. I want your answer to be 100% correct, so if you have the slightest doubt, tell me you're not sure.\n"
    "answer should be ped if there's ANY part of human inside the bounding box, otherwise fa. only answer one of these specific phrases: ped/fa/maybe\n"
    "Also provide a confidence score between 0 and 100, indicating how confident you are in your answer.\n"
    "The final answer should be in the format: ped CONF, fa CONF, maybe CONF etc.\n"
    #####################

    # "Based on the images and the description provided, is there a real human, or a part of a real human in the red bounding box?\n"
    # "I'm specifically interested in knowing whether there's a real human, or some part of a real human, INSIDE the actual red bounding box.\n"
    # "If there are humans right next to the bounding box, but the box itself does not contain any part of a human, then the answer should be NO.\n"
    # "Note that the person might be occluded or partially visible, but there still is a person inside the bounding box even if the person is occluded or partially visible.\n"
    # "Do not confuse with objects that are outside the red bounding box.\n"
    # "Note that the person might be riding a bicycle, a motorcycle, or sitting inside a car.\n"
    # "Be cautious not to confuse non-human objects, such as bicycles, motorcycles, poles, signs, bags, shadows, or reflections, for parts of a human.\n"

    # "answer should be ped if there's ANY part of human inside the bounding box, or fa is there isn't. only answer one of these specific phrases: ped or fa.\n"
    )

    user_message_content2 = [{"type": "text", "text": prompt2}] + [
        # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}", "detail": "high"}}
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        for encoded_image in encoded_images
    ]

    # Create the completion request for the second prompt
    completion2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_message_content2}  
        ],
        # temperature=0.2,
    )

    # Extract the decision from the second response
    response = completion2.choices[0].message.content
    response = response.lower()
    # print("Decision from Prompt 2:", decision_response)
    ################################################

    if 'ped' in response:
        decision = 'ped'
    elif 'fa' in response:
        decision = 'fa'
    elif 'maybe' in response:
        decision = 'maybe'
    else:
        print(f"Error: GPT-4 response is not 'ped' or 'fa'. Response: {response}")
        decision = 'maybe'
    
    confidence = extract_integer(response)
    # print(f"Decision: {decision}, Confidence: {confidence}")

    return decision, confidence

def process_images_in_folder(main_folder: str, hard_images_path: str, use_only_hard_images: bool, model: str):
    with open(hard_images_path, 'r') as f:
        hard_images = f.readlines()
    hard_images = set([x.strip() for x in hard_images])

    results = []

    for subfolder in os.listdir(main_folder):
        if not subfolder.startswith('crops_with_bb'):
        # if not subfolder.startswith('crops_with_bb_false_ManualFalse'):
            continue
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
        det_nums = [x.split('_')[-1] for x in images]

        images = sorted(images)
        # take only hard images
        if use_only_hard_images:
            for i, image_file in enumerate(images):
                if det_nums[i] not in hard_images:
                    images[i] = None
            images = [x for x in images if x is not None]
        images = images[:2]
        ##################################    

        for i, image_file in enumerate(tqdm(images, desc="Processing images", unit="image")):
            image_path = os.path.join(subfolder_path, image_file)
            # if 'det_7269' not in image_path:
            #     continue

            object_class_name, conf = get_object_class_name(image_path, model)
            results.append({
                # "image_path": image_path + "_orig_with_bb.png",
                "image_path": image_path + "_context_10.png",
                "initial_pred": initial_pred,
                "manual_gt": manual_gt,
                "gpt_pred": object_class_name,
                "confidence": conf
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
        if not subfolder.startswith('crops_with_bb'):
            continue
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
    main_folder = "/Users/tomercohen/Downloads/crop2vec_chatgpt/Manual_Tagged_crops_multi_context"
    error_folder = "/Users/tomercohen/Downloads/crop2vec_chatgpt/Manual_Tagged_ChatGPT_errors"
    hard_images_path = '/Users/tomercohen/Downloads/crop2vec_chatgpt/hard_images_Manual_Tagged_ChatGPT_errors_multi_context_all_best_f1_086.txt'
    # hard_images_path = '/Users/tomercohen/Downloads/crop2vec_chatgpt/hard_images_maybe_images.txt'
    use_only_hard_images = True
    model = "o1-preview"

    # Create an empty clone of the input directories
    create_error_directory_structure(main_folder, error_folder)

    results = process_images_in_folder(main_folder, hard_images_path, use_only_hard_images, model)
    print(f'unique gpt preds: {Counter([result["gpt_pred"] for result in results])}')

    # save results to file
    with open(f'{main_folder}/results.pkl', 'wb') as f:
        pickle.dump(results, f)

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
