import json
import os

# Directory containing the JSON files
input_dir = 'DL2_InstanceDiffusion/llm_submodule/chatgpt_data'
output_file = 'DL2_InstanceDiffusion/llm_submodule/chatgpt/bounding_boxes_chatgpt.json'

# Prepare to collect all bounding box data
all_bboxes = []

# Loop over the first 20 JSON files
for i in range(1, 21):
    file_path = os.path.join(input_dir, f'chatgpt_descriptions_bboxes{i}.json')
    
    # Check if the file exists
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            # Collect bounding boxes and category names
            bounding_boxes = [{'bbox': anno['bbox'], 'category_name': anno['category_name']} for anno in data['annos']]
            # Append the formatted data to the list
            all_bboxes.append({'description_id': i, 'bboxes': bounding_boxes})
    else:
        print(f'File {i} does not exist.')

# Save the collected bounding boxes to a new JSON file
with open(output_file, 'w') as outfile:
    for bbox_data in all_bboxes:
        json.dump(bbox_data, outfile)
        outfile.write('\n')  # Write each entry on a new line