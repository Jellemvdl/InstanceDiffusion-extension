import torch
from PIL import Image, ImageDraw
import re
from transformers import AutoModelForCausalLM, LlamaTokenizer

# Assuming original dimensions W and H
W, H = 1000, 1000  

# Function to scale bounding boxes
def scale_bounding_boxes(text, width, height):
    import re
    pattern = r'\[\[(\d+),(\d+),(\d+),(\d+)(;)?\]?\]?'
    def scale(match):
        x0, y0, x1, y1, sep = map(lambda x: int(x) if x and x.isdigit() else x, match.groups())
        # Scale coordinates
        x0_scaled = int((x0 / W) * width)
        y0_scaled = int((y0 / H) * height)
        x1_scaled = int((x1 / W) * width)
        y1_scaled = int((y1 / H) * height)
        # Format scaled coordinates with separator if present
        return f"[[{x0_scaled},{y0_scaled},{x1_scaled},{y1_scaled}{';' if sep else ''}]]"

    # Replace all bounding boxes in the text
    scaled_text = re.sub(pattern, scale, text)
    return scaled_text

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
    for i, box in enumerate(boxes):
        print(f"Drawing box: {box}")  # Debugging statement
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=(0, 0, 0), width=2)
        if descriptions is not None:
            draw.text((box[0], box[1]), descriptions[i], fill="black")
    if caption is not None:
        draw.text((0, 0), caption, fill=(255, 102, 102))
    return image    

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-grounding-generalist-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()

query = "Can you provide a global caption and detailed instance-level descriptions with bounding boxes for the image's major instances, including each instance's attributes and precise location with coordinates [[x0,y0,x1,y1]], normalized for an image size of 512 pixels width and 512 pixels height?"
image = Image.open('/home/scur0403/DL2_InstanceDiffusion/OUTPUT/gc7.5-seed0-alpha0.8/70_xl_s0.4_n20.png').convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query=query, images=[image])
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    decoded_output = tokenizer.decode(outputs[0])  # Decode the output tensor to a string
    #print(decoded_output)
    boxes, descriptions = extract_and_scale_bounding_boxes(decoded_output, 1000, 1000)
    caption = decoded_output.split('.')[0]  # Use the first sentence as the caption
    image_with_boxes = draw_boxes(boxes, descriptions, caption)

    # Save or display the image
    image_with_boxes.save("DL2_InstanceDiffusion/chatgpt/output_image.png")
