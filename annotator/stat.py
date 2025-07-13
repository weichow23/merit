'''
Copyright 2025 The MERIT Team. All rights reserved.
'''
import os
import random
from collections import Counter, defaultdict
from termcolor import cprint
from tqdm import tqdm
import numpy as np
from annotator.utils import get_attr, normalize_string, save_json_data
import matplotlib.pyplot as plt

def count_and_print_stats(data, print_flag=True, norm_value:bool=False, plot_flag=False):
    """
    Count and print the key and value count results
    :param data: input JSON data list
    :return: key_counter, value_counters
    """
    key_counter = Counter()  # Count the number of times each key appears
    value_counters = defaultdict(Counter)  # Stores the value counter corresponding to each key

    # Traverse the data and count the key and value
    for item in data:
        try:
            attr = get_attr(item)
            for key in attr.keys():
                if norm_value:
                    key = normalize_string(key)
                key_counter[key] += 1
            for key, value in attr.items():
                if isinstance(value, str):
                    if norm_value:
                        key = normalize_string(key)
                        value = normalize_string(value)
                    value_counters[key].update([value])
                else:
                    if norm_value:
                        key = normalize_string(key)
                        values = [normalize_string(v) for v in value]
                    else:
                        values = value
                    value_counters[key].update(values)
        except:
            continue

    if print_flag:
        for key, count in key_counter.most_common():
            cprint(f"{key}: {count}", 'red')

        for key in value_counters:
            count = len(value_counters[key])
            cprint(f"{key}: {count} unique values", 'cyan')
            for value, cnt in value_counters[key].items():
                cprint(f" {value}: {cnt} times", 'blue')

    if plot_flag:
        os.makedirs('figs', exist_ok=True)
        # Draw a Key Occurrences bar chart
        if key_counter:
            plt.figure(figsize=(10, 6))
            keys = list(key_counter.keys())
            counts = list(key_counter.values())
            plt.bar(keys, counts, color='skyblue')
            plt.xticks(rotation=45, ha='right', fontsize=3)
            plt.title('Key Occurrences')
            plt.tight_layout()
            plt.savefig('figs/key_occurrences.png', dpi=500)

        # Draw a Unique Value Counts histogram
        if value_counters:
            plt.figure(figsize=(10, 6))
            keys = list(value_counters.keys())
            counts = [len(value_counters[key]) for key in keys]

            plt.bar(keys, counts, color='lightgreen')
            plt.xticks(rotation=45, ha='right', fontsize=3)
            plt.title('Unique Value Counts')
            plt.tight_layout()
            plt.savefig('figs/unique_value_counts.png', dpi=500)

    return key_counter, value_counters

def language_counter(data, print_flag=True, plot_flag=False):
    """
    Count the number of products in different languages
    """
    language_counts = Counter()
    
    for item in tqdm(data):
        language = item.get('language', 'unknown')
        language_counts[language] += 1
	
    if print_flag:
        cprint("Count of commodities in each language:", 'red')
        for language, count in language_counts.most_common():
            cprint(f"{language}: {count} items", 'blue')
    
    if plot_flag:
        try:
            import matplotlib.pyplot as plt
            top_languages = dict(language_counts.most_common(15))
            
            plt.figure(figsize=(12, 8))
            plt.bar(top_languages.keys(), top_languages.values(), color='skyblue')
            plt.title('Product Language Distribution (Top 15)')
            plt.xlabel('Language')
            plt.ylabel('Product Quantity')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        except ImportError:
            cprint("Please install matplotlib to enable plotting capabilities", 'yellow')
    
    return language_counts

def country_counter(data, print_flag=True, plot_flag=False):
    """
    Count the number of products in different languages
    """
    language_counts = Counter()
    
    for item in tqdm(data):
        language = item.get('country', 'unknown')
        language_counts[language] += 1
	
    if print_flag:
        cprint("Count of commodities in each language:", 'red')
        for language, count in language_counts.most_common():
            cprint(f"{language}: {count} items", 'blue')
    
    if plot_flag:
        try:
            import matplotlib.pyplot as plt
            top_languages = dict(language_counts.most_common(15))
            
            plt.figure(figsize=(12, 8))
            plt.bar(top_languages.keys(), top_languages.values(), color='skyblue')
            plt.title('Product Language Distribution (Top 15)')
            plt.xlabel('Language')
            plt.ylabel('Product Quantity')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        except ImportError:
            cprint("Please install matplotlib to enable plotting capabilities", 'yellow')
    
    return language_counts

def filter_by_language_rates(data, sampling_rates=None):
    """
    Filter data list based on specified sampling rates for each language.
    
    :param data: List of items with 'language' field
    :param sampling_rates: Dictionary mapping language to sampling rate (0.0-1.0)
    :return: Filtered list
    """
    if sampling_rates is None:
        # Default sampling rates as specified
        sampling_rates = {
            'id': 0.2,   # 30% sampling for id
            'vi': 0.44,   # 80% sampling for vi
            'th': 0.45,   # 90% sampling for th
            'en': 1,   # 90% sampling for en
            'ms': 0.85,   # 90% sampling for ms
            'default': 0.9  # Default for any other languages
        }
    
    filtered_data = []
    for item in data:
        language = item.get('language', 'unknown')
        rate = sampling_rates.get(language, sampling_rates.get('default', 1))
        if random.random() < rate:
            filtered_data.append(item)
    
    return filtered_data

def attr_filter_a_key(data, output_path=None):
    """
    初始规则过滤 满足规则的key
    """
    print('Rest Data before filtering:', len(data))

    key_counter, value_counters = count_and_print_stats(data, print_flag=False)
    filtered_keys = {key: count for key, count in key_counter.items() if count > 100}
    filtered_unique_counts = {key: len(value_counters[key]) for key in value_counters if 
                              (len(value_counters[key]) > 2 and key in list(filtered_keys.keys()))} # There are more than 2 types of value
    filtered_keys = {key: filtered_keys[key] for key in list(filtered_unique_counts.keys())}

    # Only keep the keys in filtered_keys
    filtered_data = []
    for item in tqdm(data):
        attr = get_attr(item)
        updated_attr = {}
        for key, values in attr.items():
            if key not in list(filtered_keys.keys()):
                continue

            updated_attr[key] = values

        if updated_attr:
            item['attribute'] = updated_attr
            filtered_data.append(item)
    
        if output_path is not None:
            save_json_data(output_path, filtered_data)
    
    print('Rest Data after filtering:', len(filtered_data))

    return filtered_data

def attr_filter_a_value(data, output_path=None):
    """
    Initial rule filtering, filtering the values ​​that meet the conditions. 
    It must be written separately from the key, because the value statistics will change after the key is filtered. 
    If the two are written together, the data will be reduced, because it is equivalent to fewer 
        keys that meet the conditions at the same time.
    """
    print('Rest Data before filtering:', len(data))

    key_counter, value_counters = count_and_print_stats(data, print_flag=False)

    filtered_data = []
    for item in tqdm(data):
        attr = get_attr(item)
        updated_attr = {}
        for key, values in attr.items():
            if isinstance(values, list):
                raise ValueError("Values must be a list or tuple.")
            # Only keep the value with a number of occurrences greater than 1. 
            # Otherwise, the value is meaningless; too few may be a long tail
            if value_counters[key][values] > 4:  
                updated_attr[key] = values

        if updated_attr:
            item['attribute'] = updated_attr
            filtered_data.append(item)
    
    if output_path is not None:
        save_json_data(output_path, filtered_data)

    print('Rest Data after filtering:', len(filtered_data))
    # count_and_print_stats(filtered_data, print_flag=True)
    return filtered_data


def analyze_query_distribution(data, print_flag=True, plot_flag=False, plot_details_flag=False):
    """
    Count the distribution of keys and values in the query
    :param data: input data list, each element contains the query field
    :return: key_counter, value_counters
    """
    key_counter = Counter()
    value_counters = defaultdict(Counter)
    
    for item in data:
        query = item.get("query", {})
        for product_id, attribute_obj in query.items():
            for key, value in attribute_obj.items():
                key_counter[key] += 1
                if isinstance(value, list):
                    value_counters[key].update(value)
                else:
                    value_counters[key][value] += 1
    
    if print_flag:
        print("Query key distribution:")
        for key, count in key_counter.most_common():
            print(f"{key}: {count}")
    
    if plot_flag:
        os.makedirs('figs', exist_ok=True)
    
        # 绘制key分布图
        plt.figure(figsize=(12, 6))
        keys = [k for k, _ in key_counter.most_common()]
        counts = [c for _, c in key_counter.most_common()]
        
        plt.bar(range(len(keys)), counts, color='skyblue')
        plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
        plt.title('Distribution of Query Keys')
        plt.xlabel('Key')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('figs/query_key_distribution.png', dpi=300)
        
        # Draw the value distribution of each key
        if plot_details_flag:
            for key, counter in value_counters.items():
                if len(counter) > 0:
                    plt.figure(figsize=(12, 6))
                    values = [v for v, _ in counter.most_common(20)]  # Take the top 20 most common values
                    counts = [c for _, c in counter.most_common(20)]
                    
                    plt.bar(range(len(values)), counts, color='lightgreen')
                    plt.xticks(range(len(values)), values, rotation=45, ha='right')
                    plt.title(f'Distribution of Values for Key: {key}')
                    plt.xlabel('Value')
                    plt.ylabel('Count')
                    plt.tight_layout()
                    plt.savefig(f'figs/query_value_distribution_{key.replace(" ", "_").replace("/", "-")}.png', dpi=300)
        
        # Plot the number of unique values ​​for each key
        plt.figure(figsize=(12, 6))
        keys = list(value_counters.keys())
        unique_counts = [len(value_counters[k]) for k in keys]
        
        # Sort by number of unique values
        sorted_indices = np.argsort(unique_counts)[::-1]
        sorted_keys = [keys[i] for i in sorted_indices]
        sorted_counts = [unique_counts[i] for i in sorted_indices]
        
        plt.bar(range(len(sorted_keys)), sorted_counts, color='salmon')
        plt.xticks(range(len(sorted_keys)), sorted_keys, rotation=45, ha='right')
        plt.title('Number of Unique Values per Key')
        plt.xlabel('Key')
        plt.ylabel('Unique Value Count')
        plt.tight_layout()
        plt.savefig('figs/query_unique_value_counts.png', dpi=300)
    
    return key_counter, value_counters
