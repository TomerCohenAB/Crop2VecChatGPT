import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()
client = OpenAI(api_key=os.getenv('CHATGPT_API_KEY'))

def get_frame_instance_numbers(image_dir):
    """
    Scans the image directory for files matching the pattern '*_inst*_original.jpg'
    and extracts the frame numbers and instance numbers.
    Returns a list of tuples (frame_num, instance_num).
    """
    files = os.listdir(image_dir)
    frame_instance_nums = set()
    for file in files:
        if file.startswith('.'):
            continue  # Skip hidden files
        if file.endswith('_original.jpg'):
            # Expected format: frameNum_instInstanceNum_original.jpg
            base_name = file[:-len('_original.jpg')]  # Remove '_original.jpg' suffix
            if '_inst' in base_name:
                frame_part, inst_part = base_name.split('_inst', 1)
                frame_num = frame_part
                instance_num = inst_part
                if frame_num.isdigit() and instance_num.isdigit():
                    frame_instance_nums.add((frame_num, instance_num))
                else:
                    print(f"Warning: Non-numeric frame or instance number in filename '{file}'")
            else:
                print(f"Warning: Unexpected filename format '{file}'")
    return sorted(frame_instance_nums)

def get_object_class_name(frame_num: str, instance_num: str, image_dir: str, model: str) -> str:
    def encode_image(im_path):
        with open(im_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Construct image paths
    base_filename = f"{frame_num}_inst{instance_num}"
    original_image_path = os.path.join(image_dir, f"{base_filename}_original.jpg")
    mask_image_path = os.path.join(image_dir, f"{base_filename}_mask.jpg")
    overlay_image_path = os.path.join(image_dir, f"{base_filename}_overlay.jpg")

    # Check if all images exist
    missing_files = []
    for path in [original_image_path, mask_image_path, overlay_image_path]:
        if not os.path.exists(path):
            missing_files.append(path)
    if missing_files:
        print(f"Missing files for frame {frame_num}, instance {instance_num}: {missing_files}")
        return f"Error: Missing images for frame {frame_num}, instance {instance_num}"

    # Encode images
    encoded_images = []
    for im_path in [original_image_path, mask_image_path, overlay_image_path]:
        encoded_images.append(encode_image(im_path))

    # Prepare the prompt
    prompt = (
        "You will receive three images:\n\n"
        "1. Original Image: The original scene without any modifications.\n"
        "2. Mask Image: A black and white mask highlighting an object from the original image.\n"
        "3. Overlay Image: The original image with a semi-transparent red overlay on the object.\n\n"
        "Task:\n\n"
        "- Identify the class of the object highlighted in the images.\n"
        "- If the object is one of the Cityscapes objects listed below, provide its name.\n"
        "- If not, either name the object explicitly or state \"undefined\".\n\n"
        "Your final answer should be only one of these specific phrases:\n\n"
        "1. Cityscapes - CITYSCAPES_OBJECT_NAME\n"
        "2. Unknown\n"
        "3. OOD - NAME_OF_THE_OBJECT\n\n"
        "Cityscapes objects:\n\n"
        "- Road\n"
        "- Sidewalk\n"
        "- Building\n"
        "- Wall\n"
        "- Fence\n"
        "- Pole\n"
        "- Traffic Light\n"
        "- Traffic Sign\n"
        "- Vegetation\n"
        "- Terrain\n"
        "- Sky\n"
        "- Person\n"
        "- Rider\n"
        "- Car\n"
        "- Truck\n"
        "- Bus\n"
        "- Train\n"
        "- Motorcycle\n"
        "- Bicycle"
    )

    # Prepare the user message content with images
    user_message_content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        for encoded_image in encoded_images
    ]

    # Create the completion request
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_message_content}
        ],
    )

    # Extract the assistant's reply
    response = completion.choices[0].message.content.strip()

    return response

def overlay_text_on_image(image_path, text, output_path):
    # Open the image
    image = Image.open(image_path)

    # Create a drawing context
    draw = ImageDraw.Draw(image)
    # font = ImageFont.load_default()

    x = 0
    y = 10
    image_area = image.size[0] * image.size[1]
    if image_area < 20000:
        font_size = 10
    elif image_area < 500000:
        font_size = 20
    else:
        font_size = 40

    print(f'image shape: {image.size}, image area: {image.size[0] * image.size[1]}')

    # Draw text with a black outline for better visibility
    outline_color = 'black'
    # Draw the text multiple times slightly offset to create an outline effect
    for adj in [-2, -1, 0, 1, 2]:
        draw.text((x+adj, y), text, fill=outline_color, font_size=font_size)
        draw.text((x, y+adj), text, fill=outline_color, font_size=font_size)

    # Draw the text in white on top
    draw.text((x, y), text, fill='white', font_size=font_size)

    # Save the image
    image.save(output_path)

if __name__ == "__main__":
    image_dir = "/Users/tomercohen/Downloads/agp_chatgpt/forchat"
    output_dir = "/Users/tomercohen/Downloads/agp_chatgpt/predictions"  # Corrected spelling
    model = "gpt-4o"  # Use the appropriate model name

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Automatically find all frame and instance numbers present in image_dir
    frame_instance_nums = get_frame_instance_numbers(image_dir)

    for frame_num, instance_num in frame_instance_nums:
        class_prediction = get_object_class_name(frame_num, instance_num, image_dir, model)
        print(f"Frame {frame_num}, Instance {instance_num}: {class_prediction}")

        if class_prediction.startswith("Error:"):
            continue  # Skip frames with errors

        # Overlay the prediction text on the original image
        base_filename = f"{frame_num}_inst{instance_num}"
        original_image_path = os.path.join(image_dir, f"{base_filename}_overlay.jpg")
        output_image_path = os.path.join(output_dir, f"{base_filename}_prediction.jpg")
        overlay_text_on_image(original_image_path, class_prediction, output_image_path)
