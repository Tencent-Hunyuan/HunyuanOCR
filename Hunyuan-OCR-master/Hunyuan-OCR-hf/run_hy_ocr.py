from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration
from PIL import Image
import numpy as np
import requests
import torch

import base64
import requests
from io import BytesIO

def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n<8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]  

    return text

def get_image(input_source):
    if input_source.startswith(('http://', 'https://')):
        response = requests.get(input_source)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    else:
        return Image.open(input_source)

def main():
    model_name_or_path = "tencent/HunyuanOCR"
    processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
    img_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/chat-ui/tools-dark.png"
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": (
                    "检测并识别图片中的文字，将文本坐标格式化输出。"
                )},
            ],
        }
    ]
    messages = [messages1]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs = get_image(img_path)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    with torch.no_grad():
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    if "input_ids" in inputs:
        input_ids = inputs.input_ids
    else:
        print("inputs: # fallback", inputs)
        input_ids = inputs.inputs
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_texts = clean_repeated_substrings(processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ))
    print(output_texts)

if __name__ == '__main__':
    main()
