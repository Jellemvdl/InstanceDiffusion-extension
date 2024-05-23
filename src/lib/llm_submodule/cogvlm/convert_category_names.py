import json
import os

synonym_to_category = {
    'person': ['girl', 'boy', 'man', 'woman', 'kid', 'child', 'chef', 'baker', 'people', 'adult', 'rider', 'children', 'baby', 'worker', 'passenger', 'sister', 'biker', 'policeman', 'cop', 'officer', 'lady', 'cowboy', 'bride', 'groom', 'male', 'female', 'guy', 'traveler', 'mother', 'father', 'gentleman', 'pitcher', 'player', 'skier', 'snowboarder', 'skater', 'skateboarder', 'foreigner', 'caller', 'offender', 'coworker', 'trespasser', 'patient', 'politician', 'soldier', 'grandchild', 'serviceman', 'walker', 'drinker', 'doctor', 'bicyclist', 'thief', 'buyer', 'teenager', 'student', 'camper', 'driver', 'solider', 'hunter', 'shopper', 'villager'],
    'bicycle': ['bike', 'unicycle', 'minibike', 'trike'],
    'car': ['automobile', 'van', 'minivan', 'sedan', 'suv', 'hatchback', 'cab', 'jeep', 'coupe', 'taxicab', 'limo', 'taxi'],
    'motorcycle': ['scooter', 'motor bike', 'motor cycle', 'motorbike', 'moped'],
    'airplane': ['jetliner', 'plane', 'air plane', 'monoplane', 'aircraft', 'jet', 'airbus', 'biplane', 'seaplane'],
    'bus': ['minibus', 'trolley'],
    'train': ['locomotive', 'tramway', 'caboose'],
    'truck': ['pickup', 'lorry', 'hauler', 'firetruck'],
    'boat': ['ship', 'liner', 'sailboat', 'motorboat', 'dinghy', 'powerboat', 'speedboat', 'canoe', 'skiff', 'yacht', 'kayak', 'catamaran', 'pontoon', 'houseboat', 'vessel', 'rowboat', 'trawler', 'ferryboat', 'watercraft', 'tugboat', 'schooner', 'barge', 'ferry', 'sailboard', 'paddleboat', 'lifeboat', 'freighter', 'steamboat', 'riverboat', 'battleship', 'steamship'],
    'traffic light': ['street light', 'traffic signal', 'stop light', 'streetlight', 'stoplight'],
    'fire hydrant': ['hydrant'],
    'stop sign': [],
    'parking meter': [],
    'bench': ['pew'],
    'bird': ['ostrich', 'owl', 'seagull', 'goose', 'duck', 'parakeet', 'falcon', 'robin', 'pelican', 'waterfowl', 'heron', 'hummingbird', 'mallard', 'finch', 'pigeon', 'sparrow', 'seabird', 'osprey', 'blackbird', 'fowl', 'shorebird', 'woodpecker', 'egret', 'chickadee', 'quail', 'bluebird', 'kingfisher', 'buzzard', 'willet', 'gull', 'swan', 'bluejay', 'flamingo', 'cormorant', 'parrot', 'loon', 'gosling', 'waterbird', 'pheasant', 'rooster', 'sandpiper', 'crow', 'raven', 'turkey', 'oriole', 'cowbird', 'warbler', 'magpie', 'peacock', 'cockatiel', 'lorikeet', 'puffin', 'vulture', 'condor', 'macaw', 'peafowl', 'cockatoo', 'songbird'],
    'cat': ['kitten', 'feline', 'tabby'],
    'dog': ['puppy', 'beagle', 'pup', 'chihuahua', 'schnauzer', 'dachshund', 'rottweiler', 'canine', 'pitbull', 'collie', 'pug', 'terrier', 'poodle', 'labrador', 'doggie', 'doberman', 'mutt', 'doggy', 'spaniel', 'bulldog', 'sheepdog', 'weimaraner', 'corgi', 'cocker', 'greyhound', 'retriever', 'brindle', 'hound', 'whippet', 'husky'],
    'horse': ['colt', 'pony', 'racehorse', 'stallion', 'equine', 'mare', 'foal', 'palomino', 'mustang', 'clydesdale', 'bronc', 'bronco'],
    'sheep': ['lamb', 'ram', 'goat', 'ewe'],
    'cow': ['cattle', 'oxen', 'ox', 'calf', 'holstein', 'heifer', 'buffalo', 'bull', 'zebu', 'bison'],
    'elephant': [],
    'bear': ['panda'],
    'zebra': [],
    'giraffe': [],
    'backpack': ['knapsack'],
    'umbrella': [],
    'handbag': ['wallet', 'purse', 'briefcase'],
    'tie': ['bow', 'bow tie'],
    'suitcase': ['suit case', 'luggage'],
    'frisbee': [],
    'skis': ['ski'],
    'snowboard': [],
    'sports ball': ['ball'],
    'kite': [],
    'baseball bat': [],
    'baseball glove': [],
    'skateboard': [],
    'surfboard': ['longboard', 'skimboard', 'shortboard', 'wakeboard'],
    'tennis racket': ['racket'],
    'bottle': [],
    'wine glass': [],
    'cup': [],
    'fork': [],
    'knife': ['pocketknife', 'knive'],
    'spoon': [],
    'bowl': ['container'],
    'banana': [],
    'apple': [],
    'sandwich': ['burger', 'sub', 'cheeseburger', 'hamburger'],
    'orange': [],
    'broccoli': [],
    'carrot': [],
    'hot dog': [],
    'pizza': [],
    'donut': ['doughnut', 'bagel'],
    'cake': ['cheesecake', 'cupcake', 'shortcake', 'coffeecake', 'pancake'],
    'chair': ['seat', 'stool'],
    'couch': ['sofa', 'recliner', 'futon', 'loveseat', 'settee', 'chesterfield'],
    'potted plant': ['houseplant'],
    'bed': [],
    'dining table': ['table', 'desk'],
    'toilet': ['urinal', 'commode', 'lavatory', 'potty'],
    'tv': ['monitor', 'televison', 'television'],
    'laptop': ['computer', 'notebook', 'netbook', 'lenovo', 'macbook', 'laptop computer'],
    'mouse': [],
    'remote': [],
    'keyboard': [],
    'cell phone': ['mobile phone', 'phone', 'cellphone', 'telephone', 'phon', 'smartphone', 'iPhone'],
    'microwave': [],
    'oven': ['stovetop', 'stove', 'stove top oven'],
    'toaster': [],
    'sink': [],
    'refrigerator': ['fridge', 'freezer'],
    'book': [],
    'clock': [],
    'vase': [],
    'scissors': [],
    'teddy bear': ['teddybear'],
    'hair drier': ['hairdryer'],
    'toothbrush': []
}

def load_json(filename):
    """ Load JSON data from a file where each line is a separate JSON object. """
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def convert_category_names(data, synonym_to_category):
    # Create a reverse mapping from synonyms to the main category
    synonym_map = {}
    for main_category, synonyms in synonym_to_category.items():
        for synonym in synonyms:
            synonym_map[synonym] = main_category
        # Include the main category itself in the map
        synonym_map[main_category] = main_category

    # Process each entry in the data
    for entry in data:
        for bbox in entry['bboxes']:
            category_name = bbox['category_name']
            # Replace the category name with the main category if it exists in the map
            if category_name in synonym_map:
                bbox['category_name'] = synonym_map[category_name]

    return data

script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir,'cogvlm_data/cogvlm_bboxes.json')
output_json_path = os.path.join(script_dir, 'cogvlm_data')

data = load_json(file_path)
converted_data = convert_category_names(data, synonym_to_category)
output_json = os.path.join(output_json_path, 'converted_cogvlm_bboxes.json')
with open(output_json, 'w') as f:
    for entry in converted_data:
        json.dump(entry, f)
        f.write('\n')

