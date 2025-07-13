'''
Copyright 2025 The MERIT Team. All rights reserved.
'''
import json
import os
from tqdm import tqdm
from annotator.attribute import find_top100_keys, find_top100_values_for_each_key, reverse_dict_lookup
from annotator.utils import read_json_data

def save_specific_class(class_name, save_path="./open-data"):
    assert class_name in list(reverse_dict_lookup().keys())
    print(class_name)
    class_name_pool = reverse_dict_lookup()[class_name]

    base_path = "..." # NOTE: add your base path here
    files = [
        "...", "...", "..." # NOTE: add your file name here
    ]

    data = []
    for file_name in tqdm(files):
        file_path = os.path.join(base_path, file_name)
        with open(file_path, 'r') as file:
            for line in file:
                json_data = json.loads(line.strip())
                # json data look like below:
                # {'product_id': 1730563945600094952, 'country': 'US', 'mllm_extracted_pv': '{"Suitable Seasons": ["All seasons"], "Neckline": ["V-neck"], "Size": ["S", "M", "L"], "Fit Type": ["Casual"], "Sleeve Length": ["Half sleeve"], "Care Instruction": ["Machine wash cold", "Tumble dry low"], "Stretch": ["No stretch"], "Sleeve Type": ["Regular"], "Material": ["65% cotton", "35% polyester"], "Occasion": ["Everyday wear"], "Features": ["Basic style"], "Pattern/Print": ["Solid"], "Color": ["Black"], "Composition": ["65% cotton, 35% polyester"], "Quantity Per Pack": ["1"]}', 'first_name': 'Womenswear & Underwear', 'second_name': "Women's Tops", 'third_name': "Women's T-shirts", 'fourth_name': None, 'leaf_name': "Women's T-shirts", 'first_cid_ai': 601152, 'second_cid_ai': 842248, 'third_cid_ai': 601265, 'first_name_ai': 'Womenswear & Underwear', 'second_name_ai': "Women's Tops", 'third_name_ai': 'Blouses & Shirts', 'leaf_cid_ai': 601265, 'leaf_cid': 601302}
                if json_data['second_cid_ai'] in class_name_pool or json_data['third_cid_ai'] in class_name_pool:
                    data.append({
                        'product_id': json_data['product_id'],
                        'mllm_extracted_pv': json_data['mllm_extracted_pv']
                    })

    print(len(data))
    output_file = save_path + f'-{class_name}.json'
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    class_name = 'phone'  # A case, you can change this to other class name
    save_path = "./open-data"

    # step 1: Save the corresponding type of atlas data
    save_specific_class(class_name, save_path)

    # step 2: Find the top 100 keys -> filter in GPT
    data = read_json_data(f'{save_path}-{class_name}.json')
    find_top100_keys(data, key_output_path=f'{save_path}-{class_name}_key.json')

    # step 3: find the top 100 values ​​for each of the top 100 keys
    find_top100_values_for_each_key(data, key_path=f'{save_path}-{class_name}_key.json', 
                                    dict_output_path=f"{save_path}-{class_name}_dict.json")
