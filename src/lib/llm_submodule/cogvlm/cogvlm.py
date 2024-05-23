import torch
from PIL import Image, ImageDraw
import re
from transformers import AutoModelForCausalLM, LlamaTokenizer
import os
import json

# Assuming original dimensions W and H
W, H = 1000, 1000  

# # Function to scale bounding boxes
# def scale_bounding_boxes(text, width, height):
#     import re
#     pattern = r'\[\[(\d+),(\d+),(\d+),(\d+)(;)?\]?\]?'
#     def scale(match):
#         x0, y0, x1, y1, sep = map(lambda x: int(x) if x and x.isdigit() else x, match.groups())
#         # Scale coordinates
#         x0_scaled = int((x0 / W) * width)
#         y0_scaled = int((y0 / H) * height)
#         x1_scaled = int((x1 / W) * width)
#         y1_scaled = int((y1 / H) * height)
#         # Format scaled coordinates with separator if present
#         return f"[[{x0_scaled},{y0_scaled},{x1_scaled},{y1_scaled}{';' if sep else ''}]]"

#     # Replace all bounding boxes in the text
#     scaled_text = re.sub(pattern, scale, text)
#     return scaled_text

def extract_and_scale_bounding_boxes(text, original_width, original_height, target_width=512, target_height=512):
    pattern = r'\[\[(\d+),(\d+),(\d+),(\d+)(;)?\]?\]?'
    boxes = []
    descriptions = []
    matches = re.finditer(pattern, text)
    for match in matches:
        x0, y0, x1, y1, sep = map(lambda x: int(x) if x and x.isdigit() else x, match.groups())
        x0_scaled = int((x0 / original_width) * target_width)
        y0_scaled = int((y0 / original_height) * target_height)
        x1_scaled = int((x1 / original_width) * target_width)
        y1_scaled = int((y1 / original_height) * target_height)
        boxes.append([x0_scaled, y0_scaled, x1_scaled, y1_scaled])
        descriptions.append(text[:match.start()].split()[-1])  # Get the word before the bounding box as description
    return boxes, descriptions   

def draw_boxes(boxes, descriptions=None, caption=None, width=512, height=512):
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    bbox_descriptions = []
    for i, box in enumerate(boxes):
        #print(f"Drawing box: {box}")  # Debugging statement
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=(0, 0, 0), width=2)
        if descriptions is not None:
            description_text = descriptions[i]
            draw.text((box[0], box[1]), description_text, fill="black")
            bbox_descriptions.append({
                "bbox": box,
                "category_name": descriptions[i]
            })
    if caption is not None:
        draw.text((0, 0), caption, fill=(255, 102, 102))
    return image, bbox_descriptions  

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-grounding-generalist-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()

query = "Can you provide a global caption and detailed instance-level descriptions with bounding boxes for the image's major instances, including each instance's attributes and precise location with coordinates [[x1,y1,x2,y2]]?"
image = Image.open('/home/scur0403/DL2_InstanceDiffusion/llm_submodule/chatgpt_output/2024-05-19 10:12/gc7.5-seed0-alpha0.8/0.png').convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query=query, images=[image])
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

# Directory containing the images
image_dir = '/home/scur0403/DL2_InstanceDiffusion/llm_submodule/chatgpt_output/2024-05-19 10:12/gc7.5-seed0-alpha0.8'
output_dir = 'DL2_InstanceDiffusion/llm_submodule/cogvlm/cogvlm_bbox_images'
output_json_path = 'DL2_InstanceDiffusion/llm_submodule/cogvlm/cogvlm_data'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
all_bboxes_descriptions = []
# Process the first 20 images
for i in range(0, 60, 3):
    image_path = os.path.join(image_dir, f'{i}.png')
    if not os.path.exists(image_path):
        print(f"Image {image_path} does not exist.")
        continue

    image = Image.open(image_path).convert('RGB')
    inputs = model.build_conversation_input_ids(tokenizer, query=query, images=[image])
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        decoded_output = tokenizer.decode(outputs[0])  # Decode the output tensor to a string
        boxes, descriptions = extract_and_scale_bounding_boxes(decoded_output, 1000, 1000)
        caption = decoded_output.split('.')[0]  # Use the first sentence as the caption
        image_with_boxes, bbox_descriptions = draw_boxes(boxes, descriptions, caption)

        # Save the image with bounding boxes
        output_image_path = os.path.join(output_dir, f'output_image_{i}.png')
        image_with_boxes.save(output_image_path)
        print(f"Processed and saved: {output_image_path}")

        all_bboxes_descriptions.append({"description_id": i, "bboxes": bbox_descriptions})

# Save all bounding boxes and descriptions in a single JSON file
output_json = os.path.join(output_json_path, 'cogvlm_bboxes.json')
with open(output_json, 'w') as f:
    for bboxes in all_bboxes_descriptions:
        json.dump(bboxes, f)
        f.write('\n')
print(f"All bounding boxes and descriptions saved in: {output_json}")        
