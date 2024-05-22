import os
import json

def load_json(filename):
    """ Load JSON data from a file where each line is a separate JSON object. """
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data  
# Function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = w1 * h1
    box2Area = w2 * h2

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou    

cogvlm_bboxes_data = load_json("DL2_InstanceDiffusion/llm_submodule/cogvlm/cogvlm_data/converted_cogvlm_bboxes.json")  
chatgpt_bboxes_data = load_json("DL2_InstanceDiffusion/llm_submodule/chatgpt/bounding_boxes_chatgpt.json")       


iou_results = []
# Ensure that both data lists are of the same length or handle cases where they are not
min_length = min(len(cogvlm_bboxes_data), len(chatgpt_bboxes_data))

for i in range(min_length):
    cogvlm_entry = cogvlm_bboxes_data[i]
    chatgpt_entry = chatgpt_bboxes_data[i]

    cogvlm_id = cogvlm_entry['description_id']
    cogvlm_bboxes = cogvlm_entry['bboxes']
    chatgpt_id = chatgpt_entry['description_id']
    chatgpt_bboxes = chatgpt_entry['bboxes']

    # Compare each bbox in the current cogvlm_data entry with each bbox in the current chatgpt_data entry
    for cogvlm_bbox in cogvlm_bboxes:
        for chatgpt_bbox in chatgpt_bboxes:
            if cogvlm_bbox['category_name'].lower() == chatgpt_bbox['category_name'].lower():
                iou = calculate_iou(cogvlm_bbox['bbox'], chatgpt_bbox['bbox'])
                iou_results.append({
                    "description_id": cogvlm_id,
                    "pred_box": f"{cogvlm_bbox['category_name']} {cogvlm_bbox['bbox']}",
                    "gt_box": f"{chatgpt_bbox['category_name']} {chatgpt_bbox['bbox']}",
                    "iou": iou,
                    "label": cogvlm_bbox['category_name']
                })


best_iou_per_label = {}

for result in iou_results:
    key = (result['description_id'], result['label'])
    if key not in best_iou_per_label or best_iou_per_label[key]['iou'] < result['iou']:
        best_iou_per_label[key] = result

# Extract the best results from the dictionary to a list
filtered_iou_results = list(best_iou_per_label.values())

average_iou = sum(result['iou'] for result in filtered_iou_results) / len(filtered_iou_results)
print(f"Average IoU: {average_iou}")

# Save the filtered results to a JSON file
with open('DL2_InstanceDiffusion/llm_submodule/cogvlm/cogvlm_data/filtered_iou_results.json', 'w') as outfile:
    json.dump(filtered_iou_results, outfile, indent=4)

