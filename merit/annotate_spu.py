'''
Copyright 2025 The MERIT Team. All rights reserved.
'''
import json
import os
import hashlib
import random
import re
from torch.utils.data import Dataset
from lmdeploy.vl import load_image
from itertools import zip_longest
from tqdm import tqdm
from annotator.utils import GPT4, save_json_data, read_json_data
from annotator.stat import count_and_print_stats, language_counter
from annotator.attribute import verify_class_name, class_counter

FATHER_FOLDER_PATH = './data/'

def parse_sku_property(line, spu_special_mode):
    src_data = line['src_data']
    sku_images = src_data.get('sku_images', [])
    sku_ids = src_data.get('sku_ids', [])
    property_audit_view = src_data.get('property_audit_view', [])
    sku_property_list =  []
    for sku_id, sku_image, property_audit in zip_longest(sku_ids, sku_images, property_audit_view):
        sku_property = {
            'sku_id': sku_id,
            'sku_image': sku_image,
            'property_item_list': {property_audit['property_name']: property_audit['property_value_name']} \
                if property_audit is not None else None
        }
        if sku_image is not None:
            sku_property_list.append(sku_property)

    if spu_special_mode:
        for spu_image in src_data['images']:
            fake_sku_id = 999 * 10**16 + random.randint(0, 10**16 - 1)
            sku_property = {
                'sku_id': fake_sku_id,
                'sku_image': spu_image,
                'property_item_list': None
            }
            sku_property_list.append(sku_property)
    return sku_property_list


class SKUDataset(Dataset):
    def __init__(self, json_path, spu_special_mode=False):
        '''
        ::spu_special_mode is False as default, only use it for imagebank
        '''
        self.data = []
        self.spu_special_mode = spu_special_mode
        self._load_data(json_path)
                
    def __len__(self):
        return len(self.data)

    def _pack_item(self, item):
        sku_property_list = parse_sku_property(item, spu_special_mode=self.spu_special_mode)
        for sku_property in sku_property_list:
            sku_property.update({
                'spu_id': item["product_id"],
                'spu_description': item["src_data"]["description_text"],
                'first_category_id': item["first_category_id"],
                'second_category_id': item["second_category_id"],
                'third_category_id': item["third_category_id"],
                "country": item["country_code"],
                "language": item["src_data"]['Language']
            })
            self.data.append(sku_property)

    def _load_data(self, json_path):
        if json_path.endswith('.json') or json_path.endswith('.jsonl'):
            self._load_json_file(json_path)
        else:
            # If it is a directory, load all .json files in the directory
            if os.path.isdir(json_path):
                json_files = [os.path.join(json_path, f) for f in os.listdir(json_path) 
                             if (f.endswith('.json') or f.endswith('.jsonl')) and os.path.isfile(os.path.join(json_path, f))]
                
                if json_files:
                    for json_file in json_files:
                        self._load_json_file(json_file)
                else:
                    print(f"Warning: No JSON files found in directory {json_path}")
            else:
                print(f"Error: Path {json_path} is neither a valid file nor a directory")
    
    def _load_json_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        product_data = json.loads(line.strip())
                        self._pack_item(product_data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_path}, line: {line[:50]}...: {e}")
                        continue
                                        
        except FileNotFoundError:
            print(f"Error: File not found at path {file_path}")
        except Exception as e:
            print(f"Unexpected error loading data from {file_path}: {e}")
    
    def __getitem__(self, idx):
        return self.data[idx]
    

def template_prompt(attribute_card, img_attr, img_desc):
     return f"""
Objective:
Annotate clothing product attributes based on the provided inputs. Your annotations must align with the given attribute options and labeling rules.

Inputs:
List of Attributes and Options: A predefined list of attributes (e.g., "color," "sleeve length") and their valid options.
Product Image: The primary reference for annotation.
Image Attribute Table (Optional): Existing attribute descriptions for reference (may be incomplete or empty).
Product Description (Optional): Textual description of the product (may be inaccurate—verify against the image).

Labeling Rules:
Select the most accurate option for each attribute based on visible/verifiable details.
Skip an attribute if it cannot be determined from the inputs. Do not add new attributes.
Leverage the Image Attribute Table (if available):
If an attribute (or a semantically similar one) exists in the table, summarize its content as one of the predefined options.
If your summarized value isn't in the options, ensure it adheres to:
Exclusions: Omit niche/rare attributes.
No Overlap: Remove synonymous terms.

Your Output Format:
Please output your answer in json format. The answer is a dict, where the key is the attribute and the value is the value of the attribute. The answer should be in English.

List of Attributes and Options is
{attribute_card}

Product Image is
<image>

Image Attribute Table is
{img_attr}

Product Description is
{img_desc}
     """

def step_1_annotate(dataset, class_name, en_only_flag=True, limit_number=100):
    def load_image_for_item(item):
        return load_image(item["sku_image"])
    
    def generate_prompt_for_item(item):
        with open(f'./attribute_cards/v2_{class_name}_dict.json', 'r', encoding='utf-8') as file:
            attribute_card = json.load(file)
        return template_prompt(
            attribute_card=str(attribute_card),
            img_attr=item['property_item_list'],
            img_desc=item['spu_description']
        )
    
    def handle_result(item, response):
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        cleaned_response = json_match.group(0)
        response_dict = json.loads(cleaned_response)
        item.update({'attribute': response_dict})

    model = GPT4(max_workers=8, qps=4)

    filtered_items = []
    for item in dataset:
        if item['property_item_list'] is not None:
            # Statistics show that None and empty cases should be removed
            if item['language'] not in ['id', 'th', 'vi', 'en', 'ms']:
                continue
            # Only consider the case of single language
            if en_only_flag and item['language'] != 'en':
                continue
            # Only keep this type
            if not verify_class_name(
                id1=item['second_category_id'],
                id2=item['third_category_id'],
                target_class_name=class_name
            ):
                continue
            # Limit to 10,000 items
            filtered_items.append(item)
            if len(filtered_items) > limit_number:
                break

    # Process all items in parallel
    processed_items = model.process_batch(
        filtered_items,
        image_loader_func=load_image_for_item,
        prompt_generator_func=generate_prompt_for_item,
        result_handler_func=handle_result,
        show_progress=True
    )
    
    save_json_data(f'{FATHER_FOLDER_PATH}/v3_{class_name}.json', processed_items)

def step2_auto_fliter(json_path, class_name):
    annotate_datase = read_json_data(json_path)
    key_counter, value_counters = count_and_print_stats(annotate_datase, print_flag=False)

    filtered_data = []
    with open(f'./attribute_cards/v2_{class_name}_dict.json', 'r', encoding='utf-8') as file:
        attribute_card =  json.load(file)

    for item in tqdm(annotate_datase):
        attr = item['attribute']
        new_item = item.copy()
        filtered_attr = {}

        save_directory = f'./sku_images/{class_name}' # NOTE: path to save image
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        url = item["sku_image"]
        filename = hashlib.md5(url.encode()).hexdigest() + ".jpg"
        local_image_path = os.path.join(save_directory, filename)
        if not os.path.exists(local_image_path):
            continue
            # print(f"Downloading image to {local_image_path}")
            image = load_image(url)
            image.save(local_image_path)
        
        for key, value in attr.items():
            if len(value_counters[key])<2:
                continue

            if key in attribute_card.keys() and \
                type(value) is str and \
                value not in ['skip', 'none', 'unknown', 'not specified', value_counters[key].most_common(1)[0][0]] and \
                value_counters[key][value] > 1:
                filtered_attr[key] = value

        new_item['local_sku_image'] = local_image_path
        new_item['attribute'] = filtered_attr
        
        if new_item['attribute']!={} and os.path.exists(local_image_path):
            filtered_data.append(new_item)

    save_json_data(json_path, filtered_data)

def caption_prompt(raw_desc, attribute, lang):
    return f"""
Task: Generate an unique, expressive and accurate product title shown in Electronic shopping system for retrieval purposes.
Objective: Craft a product title using the provided original description, attributes, and pictures. Prioritize correctness and brevity by:
Incorporating key details from the attributes (mandatory).
Validating/correcting the original description against the pictures (it may be inaccurate or verbose), appropriately put the right information from original description in your title to increase richness.
Ensuring the caption is a single, fluent sentence—clear, natural, and impactful. Don't use "this" or "that" in your title. Your title can start with "a" if it is appropriate.
If the content of Attributes can be directly seen from the picture, such as color, shape, and quantity, they should not appear in the title. If the content of the Attributes cannot be seen in the image, such as material, specific size, weight and other special content, it must be reflected in the titl.

Requirements:
Uniqueness: Avoid generic phrasing; highlight distinguishing features.
Simplicity: Be succinct. Omit non-essential details while retaining key information.

Notice:
The language of your response should be consistent with the Original Description, which is {lang.upper()}.

Input:
Original Description: {raw_desc}
Attributes: {attribute}
Pictures: <image>
    """

def step_3_refine_product(product_path):
    """Process products using multi-threaded GPT-4 calls
    
    Args:
        product_path: Path to input JSON file
        output_path: Path to save processed data
        max_workers: Maximum number of worker threads
        qps: Queries per second rate limit
    """
    # Define the image loader function
    def load_image_for_item(item):
        return load_image(item["local_sku_image"])
    
    # Define the prompt generator function
    def generate_prompt_for_item(item):
        return caption_prompt(raw_desc=item['spu_description'], attribute=item.get("attribute", []), lang=item['language'])
    
    # Define the result handler
    def handle_result(item, response):
        print(response)
        item['title'] = response

    model = GPT4(max_workers=8, qps=4, api_key_id=0)
    candidates = read_json_data(product_path)
    
    # Process all items in parallel
    processed_items = model.process_batch(
        candidates,
        image_loader_func=load_image_for_item,
        prompt_generator_func=generate_prompt_for_item,
        result_handler_func=handle_result,
        show_progress=True
    )
    save_json_data(product_path.replace('v3_', 'v4_'), processed_items)

if __name__ == "__main__":
    class_name = 'gold'
    # Step 1: Label the product attributes
    # The json file in the SKUDataset, every item should like:
    # {"product_id": 1729579173357716925, "country_code": "ID", "first_category_id": 1, "second_category_id": 2, "third_category_id": 3, "cl_pay_gmv_30d": xxx, "src_data": {"size_chart_images": ["https://..."], "Country": "ID", "BizKind": "local", "seller_id": 7494561065738734013, "ProductId": 1729579173357716925, "product_status": 1, "title": "RAVEENA - Flare Pants Highwaist Cutbray - Celana Panjang Wanita", "images": ["...", "..."], "description_text": "RAVEENA - Flare Pants Highwaist Cutbray - Celana Panjang Wanita PENTING!!  KARET PINGGANG BAGIAN DALAM TIDAK DIJAHIT AGAR MODEL BAGIAN LUARNYA TERKESAN \"CLEAN LOOK\" KHUSUS UKURAN STANDART ADA LABEL M/L, NAMUN DETAIL UKURANNYA SAMA HANYA PENEMPELAN LABELNYA SAJA Terdapat 3 Ukuran : Standard (M) Jumbo (XL) Super Jumbo (XXL) Nikmati tampilan gaya retro dengan celana panjang RAVEENA Flare Pants Highwaist Cutbray. Celana berpotongan flare ini memberikan tampilan yang elegan dan modis. Polos Pinggang Tinggi Celana Panjang Flare Dirancang untuk wanita yang ingin terlihat cantik dan percaya diri, celana panjang RAVEENA Flare Pants Highwaist Cutbray memberikan kenyamanan maksimal dalam setiap gerakan. Dapat dikenakan pada semua musim, celana ini cocok untuk dipadukan dengan berbagai pilihan atasan. Dapatkan penampilan yang sempurna dengan memadukan RAVEENA Flare Pants Highwaist Cutbray dengan kemeja atau blouse favoritmu.", "description_images": [], "Categories": ["Pakaian & Pakaian Dalam Wanita", "Bawahan Wanita", "Celana Panjang"], "ProductCategoryIDs": [1, 2, 3], "package_weight": 350, "package_height": 1, "package_width": 15, "package_length": 15, "model": null, "VersionId": 136, "sku_prices": {}, "sku_images": ["…”], "sku_names": ["None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None"], "sku_ids": [1729579003305625021, 1729579003305690557, 1729579003305756093, 1729579003305821629, 1729579003305887165, 1729579003305952701, 1729579003306018237, 1729579003306083773, 1729579003306149309, 1729745387653203389, 1729745387653268925, 1729745387653334461, 1729745387653399997, 1729745387653465533, 1729745387653531069, 1730155521697809853, 1730155521697875389, 1730155521697940925, 1729579003306214845, 1729579003306280381, 1729579003306345917, 1729767929889523133, 1729767929889588669, 1729767929889654205, 1729745387653793213, 1729745387653858749, 1729745387653924285, 1729579003306411453, 1729579003306476989, 1729579003306542525, 1729745387653989821, 1729745387654055357, 1729745387654120893, 1729745387654186429, 1729745387654251965, 1729745387654317501, 1729745387654383037, 1729745387654448573, 1729745387654514109], "sku_property_names": ["Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran", "Warna", "Ukuran"], "sku_property_values": ["Black", "Standart", "Black", "Jumbo", "Black", "Super Jumbo", "BW", "Standart", "BW", "Jumbo", "BW", "Super Jumbo", "Creamy", "Standart", "Creamy", "Jumbo", "Creamy", "Super Jumbo", "Denim", "Standart", "Denim", "Jumbo", "Denim", "Super Jumbo", "Frappucino", "Standart", "Frappucino", "Jumbo", "Frappucino", "Super Jumbo", "Hazelnut", "Standart", "Hazelnut", "Jumbo", "Hazelnut", "Super Jumbo", "Ivory", "Standart", "Ivory", "Jumbo", "Ivory", "Super Jumbo", "Light Grey", "Standart", "Light Grey", "Jumbo", "Light Grey", "Super Jumbo", "Lilac", "Standart", "Lilac", "Jumbo", "Lilac", "Super Jumbo", "Mocca", "Standart", "Mocca", "Jumbo", "Mocca", "Super Jumbo", "Navy", "Standart", "Navy", "Jumbo", "Navy", "Super Jumbo", "Pink Peach", "Standart", "Pink Peach", "Jumbo", "Pink Peach", "Super Jumbo", "Stone", "Standart", "Stone", "Jumbo", "Stone", "Super Jumbo"], "warranty_period": "", "warranty_policy": "", "brand": null, "qualification_images": [], "packing_list": "", "Scene": "close", "Language": "id", "shop_name": "Raveena Grosir", "property_audit_view": [{"property_name": "Pola", "property_value_name": ["Polos"]}, {"property_name": "Musim", "property_value_name": ["Semua musim"]}, {"property_name": "Tinggi Sepinggang", "property_value_name": ["Di Atas Pinggan"]}, {"property_name": "Tipe Pakaian", "property_value_name": ["Celana Panjang Flare"]}]}}
    dataset = SKUDataset('...')  # NOTE: raw product path, please change the class and load method for your own dataset
    language_counter(dataset) # Print to see the language distribution
    class_counter(dataset) # See how many products there are in total

    product_counts = {'cloth': 109241, 'pants': 84603, 'shoes': 66165, 'headphone': 5622, 
                  'drink': 5410, 'handbag': 19581, 'phone': 18549, 'backpack': 17419, 
                  'suitcase': 8315, 'table': 2612, 'snack': 9640, 'gold': 8455, 
                  'chair': 7873, 'laptop': 1453, 'product': 1065, 'fruit': 755, 
                  'diamond': 115}
    step_1_annotate(dataset, class_name, en_only_flag=False, limit_number=product_counts[class_name])

    # Step 2: Automatically filter out untrusted attributes 
    #  + save the corresponding pictures of the products locally
    step2_auto_fliter(f'{FATHER_FOLDER_PATH}/v3_{class_name}.json', class_name)

    # Step 3: Generate product title v3 -> v4
    step_3_refine_product(f'{FATHER_FOLDER_PATH}/v3_{class_name}.json')
    # 📄 Output file: {FATHER_FOLDER_PATH}/v4_{class_name}.json