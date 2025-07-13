'''
Copyright 2025 The MERIT Team. All rights reserved.
'''
import re
import openai
from termcolor import cprint
import cv2
import json
import base64
import urllib
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from tqdm import tqdm
from ratelimiter import RateLimiter
from io import BytesIO

def get_attr(json_data):
    try:
        return json.loads(json_data['mllm_extracted_pv'])
    except:
        return json_data.get('attribute', [])
        
def normalize_string(s):
    """
	Normalize strings: lowercase, remove extra spaces and punctuation
	"""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def read_json_data(path):
	data = []
	if isinstance(path, list):
		for p in path:
			data+=read_json_data(p)
	else:
		with open(path, 'r') as file:
			for line in file:
				json_data = json.loads(line.strip())
				data.append(json_data)
	return data

def save_json_data(path, data):
	'''
	Save data in json format.
	If it is a minority language, although it is saved as an escape character, 
	it will return to the normal language when reading.
	'''
	with open(path, 'w') as file:
		for item in data:
			file.write(json.dumps(item) + '\n')

def download_image_to_base64(url: str, timeout=2, rt="str"):
    if rt == "bytes":
        try:
            req = urllib.request.urlopen(url=url, timeout=timeout)
            return base64.b64encode(req.read())
        except:
            return None

    elif rt == "str":
        try:
            req = urllib.request.urlopen(url=url, timeout=timeout)
            return base64.b64encode(req.read()).decode("utf-8")
        except:
            return None

    else:
        raise Exception(f"only support str and bytes")

class GPT4:
	def __init__(self, model_stamp='gpt-4o-2024-08-06', api_key_id=0, max_workers=4, qps=2, *args, **kwargs):	
		self.model_stamp = model_stamp
		assert model_stamp in ['gpt-4-turbo', 'gpt-4o-2024-08-06', 'gpt-4o-mini', 'o1-2024-12-17']

		api_key = [<YOUR GPT KEY>][api_key_id]  # NOTE: key in please
		if isinstance(api_key, str):
			url = "https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl"
			self.client = openai.AzureOpenAI(
				azure_endpoint=url,
				api_version=self.model_stamp,
				api_key=api_key
			)
		elif isinstance(api_key, list):
			self.client = [openai.AzureOpenAI(
				azure_endpoint= "https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl",
				api_version=self.model_stamp,
				api_key=c
			) for c in api_key]

		# Tracking token usage
		self.completion_tokens = 0
		self.prompt_tokens = 0

		# Video and image parameters
		self.test_frame = 8  # Split video into 8 frames
		self.resolution = 512

		# Threading configuration
		self.max_workers = max_workers
		self.qps = qps
		self.rate_limiter = RateLimiter(max_calls=qps, period=1)

	def _video_to_base64_frames(self, video_path, num_frames=6):
		# ref to https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
		video = cv2.VideoCapture(video_path)
		base64Frames = []
		total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
		frame_interval = max(total_frames // num_frames, 1)

		for i in range(total_frames):
			success, frame = video.read()
			if not success:
				break
			if i % frame_interval == 0:
				_, buffer = cv2.imencode(".jpg", frame)
				base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
				if len(base64Frames) >= num_frames:
					break
		video.release()
		return base64Frames

	def _replace_placeholders(self, prompt: str, images: list, video_len: int):
		img_idx = 0
		result = []

		# Split the prompt by <video> and <image> to handle each part separately
		parts = prompt.split('<video>')
		for i, part in enumerate(parts):
			if i > 0:  # if this is not the first part, it means we had a <video> placeholder
				if img_idx + video_len <= len(images):
					# Replace <video> with video_len images
					video_urls = [
						{"type": "image_url",
						"image_url": {"url": f"data:image/jpeg;base64,{images[img_idx + j]}",
									"size": self.resolution,
									"detail": "low"}}
						for j in range(video_len)
					]
					result.extend(video_urls)
					img_idx += video_len

			image_parts = part.split('<image>')
			for j, text in enumerate(image_parts):
				if j > 0:  # if this is not the first sub-part, it means we had an <image> placeholder
					if img_idx < len(images):
						image_url = {"type": "image_url",
									"image_url": {"url": f"data:image/jpeg;base64,{images[img_idx]}",
												"size": self.resolution,
												"detail": "low"}}
						result.append(image_url)
						img_idx += 1

				if text:  # Add the text part
					result.append({"type": "text", "text": text})

		return result

	def _get_response(self, client, image:list, prompt, sys_prompt=None, video_len=None):
		# ref to https://platform.openai.com/docs/guides/vision
		while True:
			try:
				if image:
					processed_prompt = self._replace_placeholders(prompt, image, video_len)
				else:
					processed_prompt = prompt
				if self.model_stamp == 'o1-2024-12-17':
					response = client.chat.completions.create(
						model=self.model_stamp,
						messages=[
							{
								"role": "user",
								"content": processed_prompt
							}
						],
						# max_completion_tokens=300,  # 300 for text; 2000 for others Not support
						# temperature=0., Not support
						seed=42,
					)
				else:
					response = client.chat.completions.create(
						model=self.model_stamp,
						messages=[
								{
									"role": "system",
									"content": sys_prompt
								},
								{
									"role"   : "user",
									"content": processed_prompt
								}
							] if sys_prompt is not None else [
								{
									"role"   : "user",
									"content": processed_prompt
								}
							],
						max_tokens=2000,  # 300 for text; 2000 for others
						temperature=0.,
						seed=42,
					)
			except openai.BadRequestError as e:
				if e.code == "sanitizer_server_error":
					continue
				elif e.code == "content_policy_violation":
					response = ""
				else:
					raise e
			except openai.InternalServerError as e:
				continue
			break
		return response

	def cost(self):
		# https://openai.com/api/pricing/
		if self.model_stamp == 'gpt-4-turbo':
			return (0.03 * self.completion_tokens + 0.01 * self.prompt_tokens) / 1000  # dollar
		elif self.model_stamp == 'gpt-4o-2024-08-06':
			return (0.005 * self.completion_tokens + 0.0015 * self.prompt_tokens) / 1000  # dollar
		elif self.model_stamp == 'gpt-4o-mini':
			return (0.00015 * self.completion_tokens + 0.000075 * self.prompt_tokens) / 1000  # dollar
		elif self.model_stamp == 'o1-2024-12-17':
			return (0.015 * self.completion_tokens + 0.0075 * self.prompt_tokens) / 1000  # dollar
		else:
			raise ValueError(f'not supporft {self.model_stamp}')

	def __call__(self, image, prompt, sys_prompt=None):
		# print(self.cost())
		v_frames = None
		if image is not None:
			base64_imgs = []
			for img in image:
				buffered = BytesIO()
				img.save(buffered, format="JPEG")
				img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
					
				base64_imgs.append(img_base64)
		else:
			base64_imgs = None
			
		# print(prompt)
		if isinstance(self.client, list):
			pointer = 0
			while True:
				client = self.client[pointer]
				try:
					response = self._get_response(client, base64_imgs, prompt, sys_prompt, len(v_frames) if v_frames is not None else None)
				except openai.RateLimitError as e:
					if pointer < len(self.client) - 1:
						pointer += 1
						continue
					else:
						raise e
				break
		else:
			response = self._get_response(self.client, base64_imgs, prompt, sys_prompt, len(v_frames) if v_frames is not None else None)
		if isinstance(response, str):
			# cprint(response, 'cyan')
			return response
		else:
			self.completion_tokens += response.usage.completion_tokens
			self.prompt_tokens += response.usage.prompt_tokens
			# cprint(response.choices[0].message.content, 'cyan')
			return response.choices[0].message.content
		
	def _process_single_item(self, item, image_loader_func, prompt_generator_func):
		"""Process a single item with the GPT model
		Args:
			item: The data item to process
			image_loader_func: Function to load image from item, should return PIL image
			prompt_generator_func: Function to generate prompt from item
			
		Returns:
			The processed item with added response
		"""
		try:
			# Load image using the provided function
			image = image_loader_func(item)
			
			# Generate prompt using the provided function
			prompt = prompt_generator_func(item)
			
			# Get model response
			response = self(image=[image] if (type(image) is not list and image) else image, prompt=prompt)
			
			# Return updated item
			return response
		except Exception as e:
			print(f"Error processing item: {e}")
			return None
	
	def process_batch(self, items, image_loader_func, prompt_generator_func, result_handler_func=None, show_progress=True):
		"""Process a batch of items using multiple threads
		
		Args:
			items: List of items to process
			image_loader_func: Function to load image for each item
			prompt_generator_func: Function to generate prompt for each item
			result_handler_func: Optional function to handle each result
			show_progress: Whether to show a progress bar
			
		Returns:
			List of processed items
		"""
		results = []
		q = Queue()
		
		# Create a thread pool
		with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
			# Submit all tasks
			futures = []
			
			if show_progress:
				items_iter = tqdm(items)
			else:
				items_iter = items
				
			for item in items_iter:
				# Apply rate limiting
				with self.rate_limiter:
					future = executor.submit(
						self._process_single_item, 
						item, 
						image_loader_func, 
						prompt_generator_func
					)
					futures.append((future, item))
					q.put(future)
			
			# Process results as they complete
			if show_progress:
				progress_bar = tqdm(total=len(futures), desc="Processing results")
			
			while not q.empty():
				future = q.get()
				try:
					response = future.result()
					
					# Find the original item for this future
					original_item = None
					for f, item in futures:
						if f == future:
							original_item = item
							break
					
					if original_item and response:
						# Apply result handler if provided, otherwise just add the response to the item
						if result_handler_func:
							result_handler_func(original_item, response)
						else:
							original_item['title'] = response
						
						results.append(original_item)
					
					if show_progress:
						progress_bar.update(1)
						
				except Exception as e:
					print(f"Error getting result: {e}")
			
			if show_progress:
				progress_bar.close()
		
		return results

def calculate_mrr(data):
    total_queries = 0
    r1, r5, r10 = 0, 0, 0
    reciprocal_ranks = []
    for item_id, rank in data.items():
        if rank == 1:
            r1 += 1
        if rank <= 5 and rank != -1:
            r5 += 1
        if rank <= 10 and rank != -1:
            r10 += 1
        if rank != -1:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)  # -1 contributes 0 to MRR
        total_queries += 1
	
    print("Rank total queries:", total_queries)
    # Compute metrics
    R1 = r1 / total_queries
    R5 = r5 / total_queries
    R10 = r10 / total_queries
    MRR10 = sum(reciprocal_ranks) / total_queries

    print(f"R@1: {R1:.4f}")
    print(f"R@5: {R5:.4f}")
    print(f"R@10: {R10:.4f}")
    print(f"MRR@10: {MRR10:.4f}")

if __name__ == "__main__":
	pass