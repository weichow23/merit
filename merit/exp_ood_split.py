import glob
from tqdm import tqdm
from annotator.attribute import class_counter, search_name_in_id_dict
from annotator.stat import language_counter, count_and_print_stats
from annotator.utils import save_json_data, read_json_data
from merit.annotate_spu import FATHER_FOLDER_PATH

def language_ood(train_queries, test_queries, train_dataset, test_dataset, product_map):
    train_language_counter_instance = language_counter(train_dataset, print_flag=True)
    test_language_counter_instance = language_counter(test_dataset, print_flag=True)

    new_train_queries = []
    for train_q_item in tqdm(train_queries):
        train_p_ids = [int(product_id['id'].split('_')[-1]) for product_id in train_q_item['passage']]
        if any(product_map[pid]['language'] in ['th', 'ms', 'id'] for pid in train_p_ids):
            continue

        new_train_queries.append(train_q_item)
    print("Train Queries", len(new_train_queries))
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/language_ood_query-train.json', new_train_queries)

    new_test_queries_th, new_test_queries_ms, new_test_queries_id = [], [], []
    for test_q_item in tqdm(test_queries):
        test_p_ids =  test_q_item['pos_candidate']+test_q_item['hard_negative']+[int(p_id) for p_id in list(test_q_item['query'].keys())]
        if all(product_map[pid]['language'] == 'th' for pid in test_p_ids):
            new_test_queries_th.append(test_q_item)
        elif all(product_map[pid]['language'] == 'ms' for pid in test_p_ids):
            new_test_queries_ms.append(test_q_item)
        elif all(product_map[pid]['language'] == 'id' for pid in test_p_ids):
            new_test_queries_id.append(test_q_item)

    print("Test (TH) Queries", len(new_test_queries_th))
    print("Test (MS) Queries", len(new_test_queries_ms))
    print("Test (ID) Queries", len(new_test_queries_id))
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/language_ood_query-test-th.json', new_test_queries_th)
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/language_ood_query-test-ms.json', new_test_queries_ms)
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/language_ood_query-test-id.json', new_test_queries_id)

def attribute_odd(train_queries, test_queries, train_dataset, test_dataset, product_map):
    attr_counter_instance, value_counter_instance = count_and_print_stats(train_dataset, print_flag=False)
    attr_counter_instance, value_counter_instance = count_and_print_stats(test_dataset, print_flag=False)

    new_train_queries = []
    for train_q_item in tqdm(train_queries):
        train_p_ids = [int(product_id['id'].split('_')[-1]) for product_id in train_q_item['passage']]
        target_attributes = ['brand', 'pattern/print', 'region of origin']
        if any(any(attr in target_attributes for attr in product_map[pid]['attribute'].keys()) for pid in train_p_ids):
            continue
        new_train_queries.append(train_q_item)
    print("Train Queries", len(new_train_queries))
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/attribute_ood_query-train.json', new_train_queries)

    new_test_queries_brand, new_test_queries_pattern, new_test_queries_region = [], [], []
    for test_q_item in tqdm(test_queries):
        all_keys = [key for attr in test_q_item['query'].values() for key in attr.keys()]
        if 'brand' in all_keys:
            new_test_queries_brand.append(test_q_item)
        elif 'pattern/print' in all_keys:
            new_test_queries_pattern.append(test_q_item)
        elif 'region of origin' in all_keys:
            new_test_queries_region.append(test_q_item)

    print("Test (brand) Queries", len(new_test_queries_brand))
    print("Test (pattern) Queries", len(new_test_queries_pattern))
    print("Test (region) Queries", len(new_test_queries_region))
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/attribute_ood_query-test-brand.json', new_test_queries_brand)
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/attribute_ood_query-test-pattern.json', new_test_queries_pattern)
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/attribute_ood_query-test-region.json', new_test_queries_region)

def get_class_name(item):
    return search_name_in_id_dict(item['second_category_id'], item['third_category_id'])

def class_odd(train_queries, test_queries, train_dataset, test_dataset, product_map):
    class_counter_instance = class_counter(train_dataset, print_flag=False)
    class_counter_instance = class_counter(test_dataset, print_flag=False)

    new_train_queries = []
    for train_q_item in tqdm(train_queries):
        train_p_ids = [int(product_id['id'].split('_')[-1]) for product_id in train_q_item['passage']]
        if any(get_class_name(product_map[pid]) in ['drink', 'table', 'phone'] for pid in train_p_ids):
            continue
        new_train_queries.append(train_q_item)
    print("Train Queries", len(new_train_queries))
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/class_ood_query-train.json', new_train_queries)

    new_test_queries_drink, new_test_queries_table, new_test_queries_phone = [], [], []
    for test_q_item in tqdm(test_queries):
        test_p_ids =  test_q_item['pos_candidate']+test_q_item['hard_negative']+[int(p_id) for p_id in list(test_q_item['query'].keys())]
        if any(get_class_name(product_map[pid]) == 'drink' for pid in test_p_ids):
            new_test_queries_drink.append(test_q_item)
        elif any(get_class_name(product_map[pid]) == 'table' for pid in test_p_ids):
            new_test_queries_table.append(test_q_item)
        elif any(get_class_name(product_map[pid]) == 'phone' for pid in test_p_ids):
            new_test_queries_phone.append(test_q_item)

    print("Test (drink) Queries", len(new_test_queries_drink))
    print("Test (table) Queries", len(new_test_queries_table))
    print("Test (phonr) Queries", len(new_test_queries_phone))
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/class_ood_query-test-drink.json', new_test_queries_drink)
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/class_ood_query-test-table.json', new_test_queries_table)
    save_json_data(f'{FATHER_FOLDER_PATH}/ood/class_ood_query-test-phone.json', new_test_queries_phone)

def find_specific_queries(query_pool, tag='train'):
    assert tag in ['train', 'test'], 'Tag must be "train" or "test"'
    product_infos = read_json_data([f'{FATHER_FOLDER_PATH}/all_products.json'])
    
    all_product_ids = []
    for query in tqdm(query_pool):
        if tag == 'train':
            all_product_ids.extend([int(product_id['id'].split('_')[-1]) for product_id in query['passage']])
        else:
            all_product_ids.extend(query['pos_candidate']+query['hard_negative']+[int(p_id) for p_id in list(query['query'].keys())])

    all_product_ids = list(set(all_product_ids))
    all_product = [product for product in product_infos if product['sku_id'] in all_product_ids]
    print('Unique Num Products ', len(all_product))
    
    # Save the product information, because generating it once is very slow
    # save_json_data(f'{FATHER_FOLDER_PATH}/ood/product-{tag}.json', all_product)
    return all_product

if __name__ == "__main__":
    train_queries = read_json_data('<Train Queries Path>')  # NOTE: add this
    test_queries = read_json_data('<Test Queries Path>')    # NOTE: add this
    
    # Seletive: save as '{FATHER_FOLDER_PATH}/ood/product-train.json'
    train_dataset = find_specific_queries(train_queries, tag='train')
    # Seletive: save as '{FATHER_FOLDER_PATH}/ood/product-test.json'
    test_dataset = find_specific_queries(test_queries, tag='test')

    product_map = {int(product["sku_id"]): product for product in train_dataset+test_dataset}

    language_ood(train_queries, test_queries, train_dataset, test_dataset, product_map)
    attribute_odd(train_queries, test_queries, train_dataset, test_dataset, product_map)
    class_odd(train_queries, test_queries, train_dataset, test_dataset, product_map)
    