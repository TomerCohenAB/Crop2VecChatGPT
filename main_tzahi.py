from openai import OpenAI
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('CHATGPT_API_KEY'))

img_with_bb: PIL.Image

buffer = BytesIO()
img_with_bb.save(buffer, format="JPEG")
buffer.seek(0)
encoded_image_with_bb = base64.b64encode(buffer.read()).decode("utf-8")

class ResponseFormat(BaseModel):
    image_description: str
    object_description: str
    object_class_name: str

response_format = ResponseFormat

prompt = (
    "I added an image, and a bounding box from the image. The bounding box contains a main car at the center of the image. "
    "I also attached the crop of the bounding box.\n"
    "Return description and values based on the following parameters:\n"

    "1. image_description - Describe the image and the scene.\n"
    "2. object_description - Describe the car that is marked in the bounding box. Be detailed in the explanation.\n"
    "3. object_class_name - the class of the object in the bounding box.\n"
)

content = [
    prompt,
    *map(lambda x: {"image": x}, [encoded_image,
                                  encoded_image_cropped,
                                  encoded_image_with_bb]),
]

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    # model="gpt-4o",
    messages=[
        {"role": "system",
         "content": "Extract data about the object marked in a bounding box in the image."},
        {"role": "user", "content": content},
    ],
    response_format=response_format,
)

gpt_responses = completion.choices[0].message.parsed
