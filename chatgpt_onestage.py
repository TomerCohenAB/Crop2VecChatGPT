from openai import OpenAI
from pydantic import BaseModel
import base64
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shutil
from collections import Counter
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('CHATGPT_API_KEY'))

postfixes = ['_orig', '_orig_with_bb', '_crop']

class ResponseFormat(BaseModel):
    object_class_name: str

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_object_class_name(image_path: str) -> str:
    postfixes_to_use = ['_orig', '_orig_with_bb', '_crop']
    encoded_images = [encode_image(image_path + p + '.png') for p in postfixes_to_use]

    image_description = (
        "There are 3 images attached. The first image is the original image, the second image is the original image "
        "with a red bounding box in the center. The third image is the crop of the red bounding box in the original image."
    )

    combined_prompt = (
        f"{image_description}\n\n"
        "Step 1: Describe in detail what is inside the red bounding box in the second image (original image with a red bounding box in the center). "
        "Mention any visible objects, shapes, textures, colors, and positions. Also, describe the immediate surroundings around the red bounding box. "
        "Be as objective and detailed as possible without making any assumptions. Avoid bias towards expecting humans ('peds') and provide a neutral description of what you see.\n\n"
        "Step 2: Based on the description from Step 1 and considering the third image (crop of the red bounding box), provide a final decision:\n"
        "- If there is any part of a real human (pedestrian) clearly inside the red bounding box, answer 'ped'.\n"
        "- If there is no part of a real human inside the red bounding box or if there is any uncertainty or doubt, answer 'fa'.\n"
        "Answer ONLY with 'ANSWER=ped' or 'ANSWER=fa' at the end of your response to avoid any ambiguity."
    )

    user_message_content = [{"type": "text", "text": combined_prompt}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}", "detail": "high"}}
        for encoded_image in encoded_images
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_message_content}]
    )

    response = completion.choices[0].message.content
    
    # Extract the final decision using the flag "ANSWER="
    final_decision = response.split("ANSWER=")[-1]
    final_decision = 'ped' if 'ped' in final_decision else 'fa'
    return final_decision

def process_images_in_folder(main_folder: str):
    results = []
    for subfolder in filter(lambda f: f.startswith("crops_with_bb"), os.listdir(main_folder)):
        subfolder_path = os.path.join(main_folder, subfolder)
        folder_parts = subfolder.split('_')

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

        images = images[:2]
        for image_file in tqdm(images, desc=f"Processing images in {subfolder}", unit="image"):
            image_path = os.path.join(subfolder_path, image_file)
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
            os.makedirs(os.path.join(error_folder, subfolder))

def copy_errors_to_directory(results, error_folder):
    for result in filter(lambda r: r['gpt_pred'] != r['manual_gt'], results):
        subfolder_name = os.path.basename(os.path.dirname(result['image_path']))
        new_image_path = os.path.join(error_folder, subfolder_name, os.path.basename(result['image_path']))
        shutil.copy(result['image_path'], new_image_path)

if __name__ == "__main__":
    main_folder = "/Users/tomercohen/Downloads/crop2vec_chatgpt/Manual_Tagged_3vars_10x"
    error_folder = "/Users/tomercohen/Downloads/crop2vec_chatgpt/Manual_Tagged_ChatGPT_errors_onestage"
    
    create_error_directory_structure(main_folder, error_folder)
    results = process_images_in_folder(main_folder)
    
    print(f'unique gpt preds: {Counter([result["gpt_pred"] for result in results])}')
    
    initial_preds = [result['initial_pred'] for result in results]
    manual_gts = [result['manual_gt'] for result in results]
    gpt_preds = [result['gpt_pred'] for result in results]
    
    calculate_metrics(manual_gts, initial_preds, "Initial Method")
    calculate_metrics(manual_gts, gpt_preds, "GPT-4 Prediction")
    copy_errors_to_directory(results, error_folder)
