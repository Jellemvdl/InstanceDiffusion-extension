from getpass import getpass
import os
import json
import time
import random

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI


OPENAI_API_KEY = getpass("Enter your API Key: ")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#create prompt template

scenes = [
    "indoor scene", 
    "outdoor scene", 
    "nature scene", 
    "commercial scene", 
    "industrial scene", 
    "transport scene", 
    "recreational scene", 
    "special event scene", 
    "agricultural scene"
]


template = """
You are an assistant that generates detailed image descriptions. Your output should include a global caption for the entire 
image and detailed instance-level descriptions with precise bounding boxes for the image's key elements.

Please follow this format:

{{
  "caption": "the caption",
  "width": 512,
  "height": 512,
  "annos": [
    {{"bbox": [x1, y1, x2, y2], "mask": [], "category_name": "some instance category name", "caption": "some instance caption"}},
    {{"bbox": [x1, y1, x2, y2], "mask": [], "category_name": "some instance category name", "caption": "some instance caption"}},
    ...
    {{"bbox": [x1, y1, x2, y2], "mask": [], "category_name": "some instance category name", "caption": "some instance caption"}},
    {{"bbox": [x1, y1, x2, y2], "mask": [], "category_name": "some instance category name", "caption": "some instance caption"}}
  ]
}}

Now, create an image description following this format, with a width of 512 pixels and a height of 512 pixels. 
The image should show an {chosen_scene}.
The instance categories should be included in this list: Person, Bicycle, Car, Motorcycle, Airplane, Bus, Train, Truck, Boat, Traffic light, Fire hydrant, Stop sign, Parking meter, Bench, Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe, Backpack, Umbrella, Handbag, Tie, Suitcase, Frisbee, Skis, Snowboard, Sports ball, Kite, Baseball bat, Baseball glove, Skateboard, Surfboard, Tennis racket, Bottle, Wine glass, Cup, Fork, Knife, Spoon, Bowl, Banana, Apple, Sandwich, Orange, Broccoli, Carrot, Hot dog, Pizza, Donut, Cake, Chair, Couch, Potted plant, Bed, Dining table, Toilet, TV, Laptop, Mouse, Remote, Keyboard, Cell phone, Microwave, Oven, Toaster, Sink, Refrigerator, Book, Clock, Vase, Scissors, Teddy bear, Hair drier, and Toothbrush.

Ensure the bounding boxes are realistic, accurately reflecting the natural sizes and arrangements of the instances. The image can show any scene you like.
"""


prompt = PromptTemplate.from_template(template)

#choose model gpt-4o and pass prompt to it
model = ChatOpenAI(model="gpt-4o")
chain = prompt | model

n_prompts = int(input("Enter the number of descriptions you would like to generate: "))

output_directory = "../chatgpt_data"
os.makedirs(output_directory, exist_ok=True)

redo = []

for i in range(n_prompts):
  file_nr = redo[i]
  chosen_scene = random.choice(scenes)
  print(chosen_scene)
  retries = 3
  for attempt in range(retries):
    try:
      response = chain.invoke({"chosen_scene": chosen_scene})  # Execute prompt with chosen_scene
      response_dict = json.loads(response.content)
      output_filename = os.path.join(output_directory, f'chatgpt_descriptions_bboxes{file_nr}.json')
      with open(output_filename, 'w') as json_file: 
        json.dump(response_dict, json_file, indent=4)
      break  #exit retry loop if successful

    except json.JSONDecodeError as e:
      print(e)
      if attempt < retries - 1:
        print("Retrying...")
        time.sleep(2)  #wait before retrying
      else:
        print(f"Failed after {retries} attempts")
    except Exception as e:
      print(f"An error occurred: {e}")
      break

print(f"Responses saved to {output_filename}")