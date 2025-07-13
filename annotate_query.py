'''
Copyright 2025 The MERIT Team. All rights reserved.
'''
import torch
import clip
import random
from termcolor import cprint
from annotator.utils import save_json_data, read_json_data, GPT4
from annotator.attribute import search_name_in_id_dict
from lmdeploy.vl import load_image
from tqdm import tqdm
from merit.annotate_spu import FATHER_FOLDER_PATH

def step1_seed_bank_rank(json_path=f"{FATHER_FOLDER_PATH}/retrieval_v3_cloth_0.5k_in_10k.json", 
                         product_path=f"{FATHER_FOLDER_PATH}/v3_cloth.json",
                         max_pos_sampling=False,
                         batch_size=4096*4): # for 80G GPU, about 1 hour
    """
    Calculate clip-i for each query candidate and then sort them. Optimized version:
    1. Preprocess all data and only store image paths
    2. De-duplicate and batch process all images
    3. Use cache to avoid repeated feature calculations
    """
    qv_pool = read_json_data(json_path)
    product_infos = read_json_data(product_path)
    product_map = {int(product["sku_id"]): product for product in product_infos}
    
    query_data = []
    all_product_ids = set() # All product IDs that need to be processed
    items_to_process = set() # Store the item index that needs to be processed
    
    for idx, item in enumerate(tqdm(qv_pool, desc="Preprocessing data")):
        q_ids = list(item["query"].keys())
        pos_ids = item["pos_candidate"]
        
        # If there are less than 2 positive samples, no sorting is required
        if len(pos_ids) < 2:
            continue
            
        all_q_exist = True
        for q_id in q_ids:
            if int(q_id) not in product_map:
                all_q_exist = False
                cprint(f"q_id {q_id} not found in product_infos", 'red')
                break
        
        if not all_q_exist:
            continue
            
        valid_pos_ids = []
        for pos_id in pos_ids:
            if int(pos_id) in product_map:
                valid_pos_ids.append(pos_id)
            else:
                cprint(f"pos_id {pos_id} not found in product_infos", 'cyan')
        
        if not valid_pos_ids:
            continue
            
        if max_pos_sampling:
            sample_size = min(64, len(valid_pos_ids))
            valid_pos_ids = random.sample(valid_pos_ids, sample_size)

        all_product_ids.update([int(id) for id in q_ids])
        all_product_ids.update([int(id) for id in valid_pos_ids])
        
        query_data.append({
            "index": idx,  # Original index
            "q_ids": q_ids,
            "pos_ids": valid_pos_ids
        })
        
        items_to_process.add(idx)
    
    if not query_data:
        cprint("No items need similarity calculation", 'yellow')
        return json_path
    
    model, transform = clip.load("ViT-B/32", "cuda")
    
    print(f"Calculate {len(all_product_ids)} product's emb...")
    product_id_to_feature = {}
    product_ids_list = list(all_product_ids)
    for i in range(0, len(product_ids_list), batch_size):
        batch_ids = product_ids_list[i:i+batch_size]
        batch_images = []
        valid_batch_ids = []
        
        # Load the pictures of this batch of products
        for pid in batch_ids:
            try:
                product = product_map[pid]
                img = load_image(product["local_sku_image"])
                batch_images.append(img)
                valid_batch_ids.append(pid)
            except Exception as e:
                cprint(f"Error loading image for product {pid}: {e}", 'red')
                continue
        
        if not batch_images:
            continue
            
        # Batch encode images
        image_inputs = torch.cat([transform(img).unsqueeze(0) for img in batch_images]).to("cuda")
        
        with torch.no_grad():
            features = model.encode_image(image_inputs).detach().cpu().float()
            
        # Store the calculation results in a dictionary
        for idx, pid in enumerate(valid_batch_ids):
            product_id_to_feature[pid] = features[idx]
    

    for query_item in tqdm(query_data, desc="Computing similarities"):
        idx = query_item["index"]
        q_ids = query_item["q_ids"]
        pos_ids = query_item["pos_ids"]
        
        query_features = torch.stack([product_id_to_feature[int(q_id)] for q_id in q_ids])
        candidate_features = torch.stack([product_id_to_feature[int(pos_id)] for pos_id in pos_ids])
        
        query_features = query_features / query_features.norm(dim=1, keepdim=True)
        candidate_features = candidate_features / candidate_features.norm(dim=1, keepdim=True)
        similarities = torch.mm(query_features, candidate_features.t()).mean(dim=0).numpy()
        
        candidates_with_scores = list(zip(pos_ids, similarities))
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        qv_pool[idx]["pos_candidate"] = [candidate[0] for candidate in candidates_with_scores]
    
    # Save the sorted results - now contains all original items
    save_json_data(json_path.replace('.json', '-rank.json'), qv_pool)
    print(f"Saved ranked results to {json_path.replace('.json', '-rank.json')}")
    print(f"Processed {len(query_data)} items out of {len(qv_pool)} total items")

    return json_path.replace('.json', '-rank.json')


def template_prompt(attribute_list, class_name):
    # Support 2 - any condition
    # case: print(template_prompt(['material', 'fit', 'collar type', 'color', 'camera resolution'], 'book'))
    # Basic prompt statement
    base_prompt = f""" Task: Generate Personalized Search Statements for Multi-Attribute Product Retrieval
Objective: Craft a concise, natural search query to retrieve a target product item by comparing conditional (reference) images and the target (desired) image. Please try not to change the original statement, but add some new attributes that are unique to the target image and not mentioned in the original statement.

Requirements:
Visual Detectability: The query must reflect attributes that are clearly visible in the images.
Personalization: Compare the products' image (input references) and the target image (desired output). Highlight differences (e.g., patterns, colors, sleeve length) and similarities (e.g., fit, material).
Simplicity: Use clear, natural language.
Independence: Your supplementary content should not directly point out what the attributes in the picture are; but the content that is not covered in the original Search Statement. you shoul imply differences/similarities naturally in the query. You need to pay special attention to this!
You must save the tag string like <Product 1> </Product 1>.

Example:
Product 1 Image: ...
Product 2 Image: ...
Target Image: ...
Search Statement: Find a T-shirt with the same color as in <Product 1> |image| </Product 1> and the same brand as in <Product 2> |image| </Product 2>.
Your Return is Find a T-shirt with the same color as in <Product 1> |image| </Product 1> and the same brand as in <Product 2> |image| </Product 2> with a small logo.
"""
    search_statement = f"Find a product of {class_name} that have the same {attribute_list[0]} with <Product 1> |image|  </Product 1>"
    
    # Add conditions based on the length of attribute_list
    for i in range(1, len(attribute_list)):
        search_statement += f" and the same {attribute_list[i]} with <Product {i+1}> |image| </Product {i+1}>"
    
    # Combine the final prompt
    final_prompt = f"""{base_prompt}

Product 1 Image: <image>"""
    
    # Add conditional images based on the length of attribute_list
    for i in range(1, len(attribute_list)):
        final_prompt += f"""
Product {i+1} Image: <image>"""
    
    final_prompt += f"""
Target Image: <image>
Search Statement: {search_statement}.
Your Return is 
    """
    return final_prompt

def step_2_annotate(json_path, product_path):
    model = GPT4(max_workers=8, qps=4)
    qv_pool = read_json_data(json_path)
    product_infos = read_json_data(product_path)
    product_map = {int(product["sku_id"]): product for product in product_infos}

    new_qv_pool = []
    
    # Prepare all items for processing
    processing_items = []
    
    for item in tqdm(qv_pool, desc="Generating qv pools"):
        pos_ids = item["pos_candidate"]
        
        # Create a processing item for each positive candidate
        if len(pos_ids) > 10:
            chosen_pos_ids = random.sample(pos_ids[:10], 3)
        elif len(pos_ids) > 3:
            chosen_pos_ids = random.sample(pos_ids, 3)
        else:
            chosen_pos_ids = pos_ids

        for pos_id in chosen_pos_ids:
            pos_product = product_map[int(pos_id)]
            if pos_product:
                processing_item = {
                    "original_item": item,
                    "pos_id": pos_id,
                    "pos_product": pos_product,
                    "attribute_list": [list(inner_dict.keys())[0] for inner_dict in item['query'].values() if inner_dict]
                }
                processing_items.append(processing_item)
            else:
                print(f"Product {pos_id} not found in product_infos.")

    # Define the image loader function
    def load_images_for_item(item):
        q_ids = list(item["original_item"]["query"].keys())
        q_images = []
        for q_id in q_ids:
            q_product = product_map[int(q_id)]
            if q_product:
                q_images.append(load_image(q_product["local_sku_image"]))
        q_images.append(load_image(item["pos_product"]["local_sku_image"]))
        return q_images
    
    # Define the prompt generator function
    def generate_prompt_for_item(item):
        return template_prompt(
            attribute_list=item["attribute_list"],
            class_name=search_name_in_id_dict(
                item["pos_product"]['second_category_id'], 
                item["pos_product"]['third_category_id']
            )
        )
    
    # Define the result handler
    def handle_result(item, response):
        q_ids = list(item["original_item"]["query"].keys())
        max_ids = len(q_ids)

        all_products_present = True
        for product_id in range(1, max_ids + 1):
            start_tag = f"<Product {product_id}>"
            end_tag = f"</Product {product_id}>"
            if start_tag not in response or end_tag not in response:
                all_products_present = False
                break
        
        
        if all_products_present:
            # Replace the |image| tag with <image>\n and the product title one by one
            modified_response = response
            image_count = modified_response.count('|image|')
            
            if image_count == max_ids:
                for i, q_id in enumerate(q_ids):
                    if '|image|' in modified_response:
                        product_title = product_map[int(q_id)]["title"]
                        replacement = f'<image>\n{product_title}'
                        modified_response = modified_response.replace('|image|', replacement, 1)

                hard_negative = item["original_item"]["pos_candidate"][:5] if len(item["original_item"]["pos_candidate"]) > 5 else \
                    item["original_item"]["pos_candidate"]

                new_qv_pool.append({
                    "query": item["original_item"]["query"],
                    "pos_candidate": [item["pos_id"]],
                    "hard_negative": list(set(hard_negative) - set([item["pos_id"]])) + item["original_item"].get("hard_negative", []),
                    'query instruction': modified_response,
                })
        else:
            print(f"Product {item['pos_id']} is not present in the response.")
    
    # Process all items in parallel
    model.process_batch(
        processing_items,
        image_loader_func=load_images_for_item,
        prompt_generator_func=generate_prompt_for_item,
        result_handler_func=handle_result,
        show_progress=True
    )
    
    # Save the results
    save_json_data(json_path.replace('-rank.json', '-instruction.json'), new_qv_pool)


if __name__ == "__main__":
    random.seed(24)
    global_json_path = f"{FATHER_FOLDER_PATH}/retrieval_v4_100k_prat2.json"
    global_product_path = f'{FATHER_FOLDER_PATH}/all_products.json'

    # Step 1: Use CLIP to reverse the candidates
    query_rank_path = step1_seed_bank_rank(json_path=f"{FATHER_FOLDER_PATH}/retrieval_v4_50k_test.json", 
                         product_path=global_product_path, max_pos_sampling=True)
    
    # Step 2: Generate search instructions
    # global_json_path is generated by ./merit/compose_query.py
    global_json_path = f"{FATHER_FOLDER_PATH}/retrieval_v4_100k_prat3.json" 
    step_2_annotate(json_path=global_json_path.replace('.json', '-rank.json'), product_path=global_product_path)
