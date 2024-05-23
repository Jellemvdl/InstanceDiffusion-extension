import torch
import clip
from PIL import Image
import os

def get_clip_score(image_path, text):
# Load the pre-trained CLIP model and the image
    model, preprocess = clip.load('ViT-B/32')
    image = Image.open(image_path)

    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score

script_dir = os.path.dirname(__file__)
image_path = os.path.join(script_dir, "chatgpt_output/2024-05-19 10:12/gc7.5-seed0-alpha0.8/72_xl_s0.4_n20.png")

text = """
A city street scene with people walking, cars parked and a sky scraper in the background. Midday light. High resolution.
"""

score = get_clip_score(image_path, text)
print(f"CLIP Score: {score}")