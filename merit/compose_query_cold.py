'''
Copyright 2025 The MERIT Team. All rights reserved.
'''
import re
import random
import torch
import clip
from tqdm import tqdm
from collections import defaultdict
from lmdeploy.vl import load_image
from annotator.utils import save_json_data, read_json_data, GPT4
from merit.annotate_spu import FATHER_FOLDER_PATH

CHOICE_NUMBER = 30

def template_prompt_diff_attr(all_attribute):
    return f"""
Task: Identify the Most Significant Attribute Difference Between Two Products

Instructions:
1. Examine the two product images carefully.
2. From the provided attribute list, select ONLY ONE attribute that represents the most significant difference between these two products. By modifying this attribute, you can convert picture 1 to picture 2
3. Even though multiple differences may exist, focus on identifying the single most visually distinct and important attribute difference. Try not to answer vague attributes like style.
4. Your response should contain ONLY the attribute name - no explanations, no quotes, no additional text.

Image 1: <image>\nImage 2: <image>\n

Available attributes to choose from:
{all_attribute}

Remember:
- Your response must be exactly one attribute from the provided list
- Do not add quotation marks or any additional text
- Select the most visually distinctive difference
- If multiple differences exist, choose the most significant one

Response (attribute name only):
"""

def template_prompt_diff_value(diff_attribute, all_values):
    return f"""
Task: Identify the Specific Value of a Differing Attribute Between Two Products

Instructions:
1. Carefully examine the two product images.
2. Focus ONLY on the following differing attribute: {diff_attribute}
3. From the provided list of possible values, select EXACTLY ONE value that best describes the {diff_attribute} that can make first product (Image 1) into product (Image 2).
4. Your response must contain ONLY the selected value - no explanations, no quotes, no additional text.
5. If none of the values in the provided list accurately describes the difference, respond with ONLY the word: no

Image 1: <image>\nImage 2: <image>\n

Differing attribute: {diff_attribute}

Available values to choose from:
{all_values}

Remember:
- Your response must be exactly one value from the provided list
- Do not add quotation marks or any additional text
- Focus exclusively on the {diff_attribute} difference between the products

Response (value only):
"""

def template_query(diff_attr, diff_value):
    return f"""
Task: Generate Enhanced Product Search Queries Based on Visual Comparison

Instructions:
1. Examine both the reference product image (Product 1) and the target product image carefully.
2. Create a natural, concise search query that helps retrieve the target product by:
   - Maintaining the original statement structure
   - Adding ONE new distinctive attribute unique to the target image
   - Including the target product's category/type in your search query

Key Requirements:
- Focus only on visually detectable attributes of specific product in both images
- Highlight meaningful differences or similarities between products
- Use natural, conversational language
- Incorporate the new attribute seamlessly without explicitly stating "this product has X attribute"
- Preserve all Product tag formatting exactly as shown: <Product 1> |image| </Product 1> and <Product 2>product description</Product 2>

Note: Product 1 will be represented by an image, while Product 2 will be represented by text attributes.

Example:
Product 1 Image: [IMAGE]
Target Image: [IMAGE]
Search Statement: Find a product with the same color as in <Product 1> |image| </Product 1> and the same brand as in <Product 2>a product with brand Nike</Product 2>.
Your Return is: Find a T-shirt with the same color as in <Product 1> |image| </Product 1> and the same brand as in <Product 2>a product with brand: Nike</Product 2> with a small logo.

Product 1 Image: <image>\nTarget Image: <image>\n

Search Statement: Find a product that has the same material as <Product 1> |image| </Product 1> and the same {diff_attr} as <Product 2>a product with {diff_attr}: {diff_value}</Product 2>.
Your Return is:
    """

def replace_product1_content(query_instruction, new_attr):
    """
    Replace content between <Product 1> and </Product 1> tags with new_attr,
    while preserving the tags themselves.
    Returns None if the pattern doesn't exist.
    """
    pattern1 = r'(<Product 1>)(.*?)(</Product 1>)'
    if re.search(pattern1, query_instruction):
        modified_query = re.sub(pattern1, r'\1' + new_attr + r'\3', query_instruction)
        return modified_query
    else:
        return None

def compose_two_condition_query():
    # Get all properties
    all_product = read_json_data(f'{FATHER_FOLDER_PATH}/all_products.json')
    attribute_values = defaultdict(set)
    for product in all_product:
        if 'attribute' in product and product['attribute']:
            for key, value in product['attribute'].items():
                attribute_values[key].add(value)
    all_attribute_and_values = {key: list(values) for key, values in attribute_values.items()}
    all_attribute = list(all_attribute_and_values.keys())
    
    qv_pool = read_json_data(f'{FATHER_FOLDER_PATH}/SimilarPair/v4_clean_annotation_mapping.json')
    global_product_path = f'{FATHER_FOLDER_PATH}/SimilarPair/v4_clean_skus.json'
    product_infos = read_json_data(global_product_path)
    product_map = {int(product["sku_id"]): product for product in product_infos}

    model = GPT4(max_workers=8, qps=4)
    
    # First pick an attribute from the attribute table
    def step_a_get_attr():
        def load_image_for_item(item):
            return [load_image(product_map[int(sku_id)]["local_sku_image"]) for sku_id in item['sku_ids']]
        
        def generate_prompt_for_item(item):
            return template_prompt_diff_attr(
                all_attribute=all_attribute,
            )
        
        def handle_result(item, response):
            if response in all_attribute:
                item.update({'attribute': response})

        # Process all items in parallel
        processed_items = model.process_batch(
            qv_pool,
            image_loader_func=load_image_for_item,
            prompt_generator_func=generate_prompt_for_item,
            result_handler_func=handle_result,
            show_progress=True
        )
        save_json_data(f'{FATHER_FOLDER_PATH}/SimilarPair/tmp.json', processed_items)
    
    # First pick an attribute from the attribute value list
    def step_b_get_value():
        saved_qv = []

        def load_image_for_item(item):
            return [load_image(product_map[int(sku_id)]["local_sku_image"]) for sku_id in item['sku_ids']]
        
        def generate_prompt_for_item(item):
            diff_attribute = item['attribute']
            return template_prompt_diff_value(
                diff_attribute=diff_attribute, 
                all_values=all_attribute_and_values[diff_attribute]
            )
        
        def handle_result(item, response):
            diff_attribute = item['attribute']
            if response in all_attribute_and_values[diff_attribute]:
                item['diff_attribute'] = {diff_attribute: response}
                saved_qv.append(item)

        # Process all items in parallel
        model.process_batch(
            read_json_data(f'{FATHER_FOLDER_PATH}/SimilarPair/tmp.json'),
            image_loader_func=load_image_for_item,
            prompt_generator_func=generate_prompt_for_item,
            result_handler_func=handle_result,
            show_progress=True
        )       

        save_json_data(f'{FATHER_FOLDER_PATH}/SimilarPair/tmp2.json', saved_qv)
    
    def step_c_get_query():
        saved_query = []

        def load_image_for_item(item):
            return [load_image(product_map[int(sku_id)]["local_sku_image"]) for sku_id in item['sku_ids']]
        
        def generate_prompt_for_item(item):
            diff_attr, diff_value = next(iter(item['diff_attribute'].items()))
            return template_query(
                diff_attr=diff_attr, diff_value=diff_value
            )
        
        def handle_result(item, response):
            modified_response = replace_product1_content(response, f'<image>\n{product_map[int(item["sku_ids"][0])]["title"]}')
            if modified_response:
                query_item = {
                    'query': {item['sku_ids'][0]: {"others": "others"}},
                    'raw_query': item['query'],
                    "pos_candidate": [item['sku_ids'][1]],
                    "hard_negative": [],
                    "query instruction": modified_response,
                    "diff_attribute": item["diff_attribute"]
                }
                saved_query.append(query_item)

        # Process all items in parallel
        model.process_batch(
            read_json_data(f'{FATHER_FOLDER_PATH}/SimilarPair/tmp2.json'),
            image_loader_func=load_image_for_item,
            prompt_generator_func=generate_prompt_for_item,
            result_handler_func=handle_result,
            show_progress=True
        )       

        save_json_data(f'{FATHER_FOLDER_PATH}/SimilarPair/tmp3.json', saved_query)

    def step_d_get_condition2(batch_size = 4096*2):
        # Add condition 2 to the labeled model. CLIP inverted 30 each. Image Bank all manually labeled
        gpt_annotate_qv_pool = read_json_data(f'{FATHER_FOLDER_PATH}/SimilarPair/tmp3.json')
        all_product_infos = read_json_data([f'{FATHER_FOLDER_PATH}/all_products.json', f'{FATHER_FOLDER_PATH}/SimilarPair/v4_clean_skus.json'])
        all_product_map = {int(product["sku_id"]): product for product in all_product_infos}

        attr_value_to_skus ={}
        for product in tqdm(all_product_infos, desc='building attribute index'):
            attribute =product.get('attribute', [])
            if attribute:
                for attr, value in attribute.items():
                    if isinstance(value, list):
                        continue
                    key = (attr, value)
                    if key not in attr_value_to_skus:
                        attr_value_to_skus[key] = [int(product['sku_id'])]
                    else:
                        attr_value_to_skus[key].append(int(product['sku_id']))
        
        all_product_ids = set()
        query_data = []
        for i, qv in enumerate(tqdm(gpt_annotate_qv_pool, desc="Collecting products")):
            try:
                diff_attr, diff_value = next(iter(qv['diff_attribute'].items()))
                cands = attr_value_to_skus.get((diff_attr, diff_value), [])
                
                if not cands:
                    continue
                    
                if len(cands) > 128:
                    cands = random.sample(cands, 128)
                
                query_id = qv["pos_candidate"][0]
                all_product_ids.add(int(query_id))
                all_product_ids.update([int(cand_id) for cand_id in cands])
                
                query_data.append({
                    "index": i,
                    "query_id": query_id,
                    "candidates": cands,
                    "qv": qv
                })
            except Exception as e:
                print(f"Error collecting products for item {i}: {e}")
        
        model, transform = clip.load("ViT-B/32", "cuda")
        
        product_id_to_feature = {}
        product_ids_list = list(all_product_ids)
        
        for i in range(0, len(product_ids_list), batch_size):
            batch_ids = product_ids_list[i:i+batch_size]
            batch_images = []
            valid_batch_ids = []
            
            for pid in batch_ids:
                try:
                    img = load_image(all_product_map[pid]['local_sku_image'])
                    batch_images.append(img)
                    valid_batch_ids.append(pid)
                except Exception as e:
                    print(f"Error loading image for product {pid}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            image_inputs = torch.cat([transform(img).unsqueeze(0) for img in batch_images]).to("cuda")
            
            with torch.no_grad():
                features = model.encode_image(image_inputs)
                features = features / features.norm(dim=1, keepdim=True)
                
            for idx, pid in enumerate(valid_batch_ids):
                product_id_to_feature[pid] = features[idx]
        
        new_gpt_annotate_qv_pool = []
        for query_item in tqdm(query_data, desc="Computing similarities"):
            idx = query_item["index"]
            query_id = int(query_item["query_id"])
            candidates = query_item["candidates"]
            qv = query_item["qv"]
            
            if query_id not in product_id_to_feature:
                continue
                
            valid_cands = []
            valid_features = []
            
            for cand_id in candidates:
                cand_id = int(cand_id)
                if cand_id in product_id_to_feature:
                    valid_cands.append(cand_id)
                    valid_features.append(product_id_to_feature[cand_id])
            
            if not valid_cands:
                continue
                
            q_feature = product_id_to_feature[query_id].unsqueeze(0)
            cand_features = torch.stack(valid_features)
            
            similarities = torch.mm(q_feature, cand_features.t()).squeeze().cpu().numpy()
            
            candidates_with_scores = list(zip(valid_cands, similarities))
            candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            top_candidates = candidates_with_scores[:min(CHOICE_NUMBER, len(candidates_with_scores))]
            qv['choice_candidate'] = [int(cand[0]) for cand in top_candidates if int(cand[0])!= int(query_id)]
            new_gpt_annotate_qv_pool.append(qv)
        
        save_json_data(f'{FATHER_FOLDER_PATH}/retrieval_v4-cold_80k_test.json', new_gpt_annotate_qv_pool)

    step_a_get_attr()
    step_b_get_value()
    step_c_get_query()
    step_d_get_condition2()

def normal_get_pos(batch_size = 4096*2):
    normal_qv_pool = read_json_data(f'{FATHER_FOLDER_PATH}/retrieval_v4_100k_prat1-instruction.json')

    all_product_infos = read_json_data([f'{FATHER_FOLDER_PATH}/all_products.json', f'{FATHER_FOLDER_PATH}/SimilarPair/v4_clean_skus.json'])
    all_product_map = {int(product["sku_id"]): product for product in all_product_infos}
    attr_value_to_skus ={}
    for product in tqdm(all_product_infos, desc='building attribute index'):
        attribute =product.get('attribute', [])
        if attribute:
            for attr, value in attribute.items():
                if isinstance(value, list):
                    continue
                key = (attr, value)
                if key not in attr_value_to_skus:
                    attr_value_to_skus[key] = [int(product['sku_id'])]
                else:
                    attr_value_to_skus[key].append(int(product['sku_id']))
    
    all_product_ids = set()
    query_data = []
    for i, qv in enumerate(tqdm(normal_qv_pool, desc="Collecting products")):
        try:
            query_conditions = []
            for query_item in qv['query'].values():
                for attr, value in query_item.items():
                    query_conditions.append((attr, value))
            all_candidates_sets = []
            for attr, value in query_conditions:
                matching_candidates = attr_value_to_skus.get((attr, value), [])
                if matching_candidates:
                    all_candidates_sets.append(set(matching_candidates))
            cands = list(set.intersection(*all_candidates_sets))
            
            if not cands:
                continue
                
            if len(cands) > 128:
                cands = random.sample(cands, 128)
            
            query_id = [int(sku_id) for sku_id in qv['query'].keys()]
            all_product_ids.update(query_id)
            all_product_ids.update([int(cand_id) for cand_id in cands])
            
            query_data.append({
                "index": i,
                "query_id": query_id,
                "candidates": cands,
                "qv": qv
            })
        except Exception as e:
            print(f"Error collecting products for item {i}: {e}")

    model, transform = clip.load("ViT-B/32", "cuda")
    product_id_to_feature = {}
    product_ids_list = list(all_product_ids)
    
    for i in range(0, len(product_ids_list), batch_size):
        batch_ids = product_ids_list[i:i+batch_size]
        batch_images = []
        valid_batch_ids = []
        
        for pid in batch_ids:
            try:
                img = load_image(all_product_map[pid]['local_sku_image'])
                batch_images.append(img)
                valid_batch_ids.append(pid)
            except Exception as e:
                print(f"Error loading image for product {pid}: {e}")
                continue
        
        if not batch_images:
            continue
            
        image_inputs = torch.cat([transform(img).unsqueeze(0) for img in batch_images]).to("cuda")
        
        with torch.no_grad():
            features = model.encode_image(image_inputs)
            features = features / features.norm(dim=1, keepdim=True)
            
        for idx, pid in enumerate(valid_batch_ids):
            product_id_to_feature[pid] = features[idx]
    
    new_gpt_annotate_qv_pool = []
    for query_item in tqdm(query_data, desc="Computing similarities"):
        idx = query_item["index"]
        query_ids = query_item["query_id"]  # List of IDs
        candidates = query_item["candidates"]
        qv = query_item["qv"]
        
        # Get valid candidates and their features
        valid_cands = []
        valid_features = []
        
        for cand_id in candidates:
            cand_id = int(cand_id)
            if cand_id in product_id_to_feature and cand_id not in query_ids:
                valid_cands.append(cand_id)
                valid_features.append(product_id_to_feature[cand_id])
        
        if not valid_cands:
            continue
        cand_features = torch.stack(valid_features)
        
        all_top_candidates = set()
        for q_id in query_ids:
            if q_id not in product_id_to_feature:
                continue
                    
            query_feature = product_id_to_feature[q_id].unsqueeze(0)
            
            # Calculate similarity between this query and all candidates
            # Use view(-1) to ensure we always get a 1-d tensor regardless of number of candidates
            similarities = torch.mm(query_feature, cand_features.t()).view(-1).cpu().numpy()
            
            # Sort candidates by similarity
            candidates_with_scores = list(zip(valid_cands, similarities))
            candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Add top candidates for this query
            query_top_candidates = candidates_with_scores[:min(CHOICE_NUMBER//2, len(candidates_with_scores))]
            all_top_candidates.update([cand[0] for cand in query_top_candidates])
        
        # Convert set to list and ensure query IDs are excluded
        top_candidates_list = [cand for cand in all_top_candidates if cand not in query_ids]    
        qv['choice_candidate'] = list(set([int(cand) for cand in top_candidates_list]))
        new_gpt_annotate_qv_pool.append(qv)
    
    save_json_data(f'{FATHER_FOLDER_PATH}/retrieval_v4_50k-instruction_test.json', new_gpt_annotate_qv_pool)

if __name__ == "__main__":
    # ------------ Processing method A ------------
    # Skip the preprocessing step. 
    # In general, you need to get two very similar but slightly different products as a pair. 
    # For details, you can refer to our paper
    compose_two_condition_query()

    # ------------ Processing method B ------------
    # Select different SKU products from the same SPU product
    normal_get_pos()