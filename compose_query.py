'''
Copyright 2025 The MERIT Team. All rights reserved.
'''
import random
import glob
import threading
import time
from tqdm import tqdm
from annotator.utils import read_json_data, get_attr, save_json_data
from annotator.stat import count_and_print_stats, language_counter, filter_by_language_rates, \
                            attr_filter_a_key, attr_filter_a_value, analyze_query_distribution
from merit.annotate_spu import FATHER_FOLDER_PATH

SAME_KEY_HOLDER = 'product style'

def construct_product_pools():
    # Step 1: Language Balance
    data = read_json_data(glob.glob(f'{FATHER_FOLDER_PATH}/v4_*'))
    language_counter(data)
    data = filter_by_language_rates(data)
    language_counter(data)
    save_json_data(f'{FATHER_FOLDER_PATH}/all_products.json', data)

    # Step 2: Filter the frequency of occurrence of key values
    data = read_json_data(f'{FATHER_FOLDER_PATH}/all_products.json')
    data = attr_filter_a_key(data)
    data = attr_filter_a_value(data, f'{FATHER_FOLDER_PATH}/all_products.json')
    language_counter(data)

def generate_query_from_items(data, num_conditions):
    """
    Extract attributes from randomly selected items to generate queries
    """
    selected_items = random.sample(data, num_conditions)
    query = {}
    condition_keys_values = []
    
    for item in selected_items:
        attr = get_attr(item)
        if not attr or not attr.items():
            return None, []

        items_list = list(attr.items())
        non_color_items = [(k, v) for k, v in items_list if k != 'color']
        if random.random()>0.8:
            if not non_color_items:
                return None, []
            key, values = random.choice(non_color_items)
        else:
            key, values = random.choice(items_list)
            
        if len(values) == 0:
            return None, []

        # Check if there is already an identical (key, values) pair
        if (key, values) in condition_keys_values:
            return None, []
        
        condition_keys_values.append((key, values))
        query[item["sku_id"]] = {key: values}
    
    conditions = []
    for item_id, attrs in query.items():
        for key, values in attrs.items():
            conditions.append((key, values))
    
    return query, conditions


def generate_query_from_attr_pool(data, attr_pool, num_conditions):
    """
    For each selected attribute key, randomly search for products that contain that attribute
    """
    selected_keys = random.sample(attr_pool, num_conditions)
    
    query = {}
    conditions = []
    for key in selected_keys:
        shuffled_indices = list(range(len(data)))
        random.shuffle(shuffled_indices)
        for idx in shuffled_indices:
            item = data[idx]
            attr = get_attr(item)
            
            if attr and key in attr and attr[key]: 
                values = attr[key]
                conditions.append((key, values))
                query[item["sku_id"]] = {key: values}
                break
       
    return query, conditions

def get_normal_pos_candidates(data, query, conditions):
    # Divide positive samples
    pos_candidates = []
    for item in data:
        attr = get_attr(item)
        # All conditions are met and not in the condition
        if all(key in attr and attr[key] == values for key, values in conditions) and item["sku_id"] not in list(query.keys()):
            pos_candidates.append(item["sku_id"])
    
    pos_rate = len(pos_candidates) / len(data)
    
    # It is very likely that you are the same, so you should delete it; at the same time, it should not be too large
    if pos_rate > 0 and pos_rate < 0.05:
        return {
            "query": query,
            "pos_candidate": pos_candidates,
            "pos_rate": pos_rate  # Positive sample ratio
        }
    else:
        return None

def generate_query_spu_first(data, spu_infos, valid_spus):
    """
    First select an item, then find another item with the same spu_id as a positive sample, 
        and then select the difference attribute as another condition
    num_conditions = 2
    """
    selected_spu = random.choice(valid_spus)
    skus_in_spu = spu_infos[selected_spu]
    item1, item2 = random.sample(skus_in_spu, 2)
    attr1 = get_attr(item1)
    attr2 = get_attr(item2)

    if not attr1 or not attr2:
        return None
    
    # Find the difference attributes, and prioritize those that are not color. 
    # Usually they are 0-3, with 1 being more common, so there is no need to go through them all
    diff_keys = []
    same_keys = [] 
    for key in attr1:
        if key in attr2: 
            if attr1[key] != attr2[key]:
                if random.random()<0.8 or key not in ['color']:
                    diff_keys.append(key)
            else:
                same_keys.append(key)

    if not diff_keys:
        return None
    diff_key = random.choice(diff_keys)

    if len(same_keys)==0:
        same_key = SAME_KEY_HOLDER # special keys
    else:
        same_key = random.choice(same_keys)
    
    # Randomly select another item that contains this difference attribute and corresponding value
    shuffled_indices = list(range(len(data)))
    random.shuffle(shuffled_indices)
    other_item = None
    for idx in shuffled_indices:
        item = data[idx]
        # If it is not the same spu, it must be not the same sku
        if item["spu_id"] != selected_spu:
            attr = get_attr(item)
            # 2 is the search target
            if attr and diff_key in attr and attr[diff_key] == attr2[diff_key]: 
                other_item = item
                break

    if not other_item:
        return None
    
    return {
        "query": {
            item1["sku_id"]: {same_key: attr1[same_key]} if same_key!=SAME_KEY_HOLDER else {SAME_KEY_HOLDER: 'as product image'},
            other_item["sku_id"]: {diff_key: attr2[diff_key]}
        },
        "pos_candidate": [item2["sku_id"]],
        "pos_rate": 1/len(data),
        "hard_negative": list(it['sku_id'] for it in skus_in_spu if it!= item1 and it!= item2),
    }

def find_spus_with_common_attribute_keys(data):
    # Get key statistics from data
    key_counter, _ = count_and_print_stats(data=data, print_flag=False)
    attr_pool = list(key_counter.keys())
    
    # Create a mapping from spu_id to items
    spu_infos = {}
    for item in data:
        spu_id = item["spu_id"]
        if spu_id not in spu_infos:
            spu_infos[spu_id] = []
        spu_infos[spu_id].append(item)
    
    # Find SPUs with the same attribute key
    new_spu_infos = {}
    for spu_id, items in spu_infos.items():
        # Only process SPUs with more than 1 item
        if len(items) <= 1:
            continue

        # Keep track of which items each property key appears in
        key_to_items = {}
        has_common_key = False
        
        # Traverse all items in SPU
        for item in items:
            # Get all attribute keys of the item
            attr_keys = item["attribute"].keys()
            
            # Check each property key
            for key in attr_keys:
                if key not in key_to_items:
                    key_to_items[key] = []
                key_to_items[key].append(item)
                
                # If the key already appears in two or more items, mark it as a common key found
                if len(key_to_items[key]) >= 2:
                    has_common_key = True
        
        # If an item with a common key is found, save it to a new map
        if has_common_key:
            # Only keep items with common keys
            common_items = []
            for key, items_with_key in key_to_items.items():
                if len(items_with_key) >= 2:
                    # Add these items to common_items to avoid duplication
                    for item in items_with_key:
                        if item not in common_items:
                            common_items.append(item)
            
            new_spu_infos[spu_id] = common_items
    
    valid_spus = list(new_spu_infos.keys())
    
    return valid_spus, new_spu_infos, attr_pool

def generate_query_and_candidates(data, num_queries=500, num_workers=32):
    """
    Multithreaded generation of combined queries and candidate sets
    num_queries is the number of queries generated, but due to multithreading, 
        the number of queries generated may slightly exceed
    """
    valid_spus, spu_infos, attr_pool = find_spus_with_common_attribute_keys(data)
    print(f"Found {len(valid_spus)} SPUs with items sharing common attribute keys")
    
    # Shared Variables
    queries = []
    lock = threading.Lock()
    pbar = tqdm(total=num_queries, desc="Generating queries")

    def worker():
        while True:
            with lock:
                if len(queries) >= num_queries:
                    break
            
            method = random.choices(['item_normal', 'attr_normal', 'spu_first'], weights=[0.3, 0.2, 0.5])[0]
            num_conditions = random.choices([2, 3, 4], weights=[0.55, 0.20, 0.25])[0]

            query_item = None
            
            if method == 'item_normal':
                query, conditions = generate_query_from_items(data, num_conditions)
                if query and len(conditions) == num_conditions:
                    query_item = get_normal_pos_candidates(data, query, conditions)
                
            elif method == 'attr_normal':
                query, conditions = generate_query_from_attr_pool(data, attr_pool, num_conditions)
                if query and len(conditions) == num_conditions:
                    query_item = get_normal_pos_candidates(data, query, conditions)
                
            elif method == 'spu_first':
                query_item = generate_query_spu_first(data, spu_infos, valid_spus)
   
            if query_item:
                with lock:
                    queries.append(query_item)
                    pbar.update(1)
    
    # Create and start a thread
    threads = []
    for _ in range(num_workers):
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # The main thread waits until enough queries are generated or all threads finish
    while len(queries) < num_queries:
        if all(not t.is_alive() for t in threads):
            break
        time.sleep(0.05)
    
    pbar.close()
    print(f"Generated {len(queries)} queries with candidates.")
    return queries

if __name__ == '__main__':
    random.seed(28)
    
    # Step 1: Build a product pool that participates in the final query construction
    # 📄 Input File from ./merit/annotate_spu.py : {FATHER_FOLDER_PATH}/v4_{class_name}.json
    construct_product_pools()
    # An item in all_products looks like:
    # {"sku_id": 1729996664698341498, "sku_image": "...", "property_item_list": {"Pattern": ["Plain"]}, "spu_id": 1729996650447145082, "spu_description": "Name:Fashion Daily Laptop Bag Wearproof Travel Backpacks Sports Backpack Material: Nylon Backpacks Type: Softback Dimension: Height: 46cm Length: 32cm Width: 15cm Shoulder Strap Size: 76cm Chest Strap Size: 29cm Gender: Unisex Function: Hiking,Camping,Traveling,Skiing,Rock Climbing,Running,Exercising ect. Color:Black,Navy Blue,Pink,Gray Note: Due to the different monitor and light effect, the actual color of the item might be slightly different from the color showed on the pictures. Thank you! Please allow 1-2cm measuring deviation due to manual measurement.", "first_category_id": 824584, "second_category_id": 902792, "third_category_id": 601446, "country": "PH", "language": "en", "attribute": {"style": "sport"}, "local_sku_image": "...", "title": "Sporty Nylon Laptop Backpack with Wearproof Design for Travel and Outdoor Activities"}

    # Step 2: Build retrieval data
    data = read_json_data(f'{FATHER_FOLDER_PATH}/all_products.json')
    # data = random.sample(data, 100000)
    queries = generate_query_and_candidates(data, num_queries=50000, num_workers=32)
    # An item in queries, looks like:
    # {"query": {"1731689392940156946": {"product style": "as product image"}, "1730019307228531271": {"color": "blue"}}, "pos_candidate": [1732018626492074002], "hard_negative": [], "query instruction": "Find a product of handbag that has the same product style with <Product 1> <image>\nT\u00fai x\u00e1ch n\u1eef da PU cao c\u1ea5p m\u00e0u burgundy v\u1edbi thi\u1ebft k\u1ebf th\u1eddi trang v\u00e0 d\u00e2y \u0111eo vai ti\u1ec7n l\u1ee3i. </Product 1> and the same color with <Product 2> <image>\nGi\u00e0y th\u1ec3 thao unisex cho b\u00e9 2-15 tu\u1ed5i, ch\u1ea5t li\u1ec7u cao c\u1ea5p, \u0111\u1ebf m\u1ec1m 3,5 cm, ki\u1ec3u d\u00e1ng n\u0103ng \u0111\u1ed9ng, bu\u1ed9c d\u00e2y. </Product 2> with a metallic finish."}
    save_json_data(f"{FATHER_FOLDER_PATH}/retrieval_v4_50k_test.json", queries)

    # (Selective) Step 3: See the key distribution
    queries = read_json_data(f"{FATHER_FOLDER_PATH}/retrieval_v4_all_0.5k_in_14k.json")
    analyze_query_distribution(queries, print_flag=True, plot_flag=True, plot_details_flag=False)
