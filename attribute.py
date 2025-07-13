'''
Copyright 2025 The MERIT Team. All rights reserved.
'''
import json
from tqdm import tqdm
from termcolor import cprint
from collections import Counter
from annotator.stat import count_and_print_stats

def find_top100_keys(data, key_output_path=None):
    '''
    Keep 10-110 keys (keep all if there are less than 10 available keys).
    '''
    key_counter, value_counters = count_and_print_stats(data, print_flag=False)
    sorted_keys = sorted(value_counters.keys(), 
                        key=lambda k: len(value_counters[k]), 
                        reverse=True)
    filtered_keys = sorted_keys[:100] if len(sorted_keys) >= 100 else sorted_keys
    if key_output_path:
        keys_info = filtered_keys
        with open(key_output_path, 'w', encoding='utf-8') as f:
            json.dump(keys_info, f, ensure_ascii=False, indent=2)

def find_top100_values_for_each_key(data, key_path, dict_output_path):
    with open(key_path, 'r', encoding='utf-8') as f:
        top_keys = json.load(f)

    # Count the frequency of occurrence of key and value
    key_counter, value_counters = count_and_print_stats(data, print_flag=False, norm_value=True)
    
    # Find the top 100 most frequent values ​​for each key
    top_key_values_dict = {}
    for key, raw_values in tqdm(top_keys.items()):
        if key in value_counters:
            # Sort by frequency in descending order and take the first 100
            top_values = sorted(value_counters[key].items(),
                              key=lambda item: item[1],
                              reverse=True)[:100]
            top_key_values_dict[key] = [value for value, count in top_values] + raw_values
        else:
            top_key_values_dict[key] = raw_values
    
    with open(dict_output_path, 'w', encoding='utf-8') as f:
        json.dump(top_key_values_dict, f, ensure_ascii=False, indent=2)

id_dict = {
    # Food & Beverage
    923912: 'fruit',    # 3 - Fruit
    922504: 'snack',    # 3 - Dried Snacks
    914824: 'drink',    # 2 - Drinks
    
    # Clothing
    842248: 'cloth',    # 2 - Women's Tops
    839944: 'cloth',    # 2 - Men's Tops
    802952: 'cloth',    # 3 - Tops
    804232: 'cloth',    # 3 - Tops
    835720: 'cloth',    # 3 - Sports Tops
    
    # Pants & Bottoms
    842376: 'pants',    # 2 - Women's Bottoms
    840072: 'pants',    # 2 - Men's Bottoms
    803208: 'pants',    # 3 - Boys' Bottoms
    803848: 'pants',    # 3 - Underwear
    
    # Footwear
    834696: 'shoes',    # 2 - Sports Footwear
    900488: 'shoes',    # 2 - Women's Shoes
    900616: 'shoes',    # 2 - Men's Shoes
    805128: 'shoes',    # 2 - Boys' Footwear
    806024: 'shoes',    # 2 - Girls' Footwear
    
    # Electronics
    602097: 'phone',    # 3 - Mobile Phones
    601990: 'headphone', # 3 - Headphones, Earphones & Accessories
    601756: 'laptop',   # 3 - Laptops
    601836: 'laptop',   # 3 - Desktop Computers
    854672: 'laptop',   # 3 - All-in-One Desktops
    
    # Bags & Luggage
    601446: 'backpack', # 3 - Backpacks
    601445: 'handbag',  # 3 - Women's Handbags
    903688: 'suitcase', # 3 - Luggage
    
    # Jewelry
    955016: 'gold',     # 2 - Gold
    955272: 'diamond',  # 2 - Diamond

    # Furniture
    875912: 'table',    # 3 - Tables & Desks
    880264: 'table',    # 3 - Tables & Desks
    876040: 'chair',    # 3 - Chairs
    879624: 'chair'     # 3 - Chairs
}

def reverse_dict_lookup():
    reversed_dict = {}
    for key, value in id_dict.items():
        if value not in reversed_dict:
            reversed_dict[value] = []
        reversed_dict[value].append(key)

    return reversed_dict

def search_name_in_id_dict(id1, id2):
    '''
    id1 is the secondary key, id2 is the tertiary key
    '''
    name1 = id_dict.get(id1, None)
    name2 = id_dict.get(id2, None)
    
    if name1 is not None:
        return name1
    elif name2 is not None:
        return name2
    else:
        return 'product'

def verify_class_name(id1, id2, target_class_name):
    target_class_list = reverse_dict_lookup().get(target_class_name, [])
    if id1 in target_class_list or id2 in target_class_list:
        return True
    else:
        return False

def class_counter(data, print_flag=True, plot_flag=False):
    """
    Count the number of products under different classes
    """
    class_counts = Counter()
    
    for item in tqdm(data):
        try:
            class_name = search_name_in_id_dict(item['second_category_id'], item['third_category_id'])
        except:
            continue
        class_counts[class_name] += 1
    
    if print_flag:
        cprint("Count of commodities in each class:", 'red')
        for class_name, count in class_counts.most_common():
            cprint(f"{class_name}: {count} Items", 'blue')
    
    if plot_flag:
        try:
            import matplotlib.pyplot as plt
            # Get the top 15 most common categories for plotting
            top_classes = dict(class_counts.most_common())
            
            plt.figure(figsize=(12, 8))
            plt.bar(top_classes.keys(), top_classes.values(), color='skyblue')
            plt.title('Product Category Quantity Distribution (Top 15)')
            plt.xlabel('Category Name')
            plt.ylabel('Product Quantity')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        except ImportError:
            cprint("Please install matplotlib to enable plotting capabilities", 'yellow')
    
    return class_counts

if __name__ == '__main__':
    print(reverse_dict_lookup().keys())