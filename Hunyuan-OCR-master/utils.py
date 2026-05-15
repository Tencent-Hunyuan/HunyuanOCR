import re
import os
import json
import random
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SPOTTING_COORD_PATTERN = re.compile(r'\((\d+),(\d+)\),\((\d+),(\d+)\)')


def _iter_spotting_items(response: str):
    """Yield text spans and coordinate groups from a spotting response."""
    text_start = 0
    for match in _SPOTTING_COORD_PATTERN.finditer(response):
        text = response[text_start:match.start()].strip()
        coord_matches = (
            (match.group(1), match.group(2)),
            (match.group(3), match.group(4)),
        )
        yield text, coord_matches, match
        text_start = match.end()


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

def norm_formula_HYOCR(gt_form,pre_form):
    """
    NORMARLIZE for OmniDocBench.input:gt_form,pred_form(from quick_match_display_formula)
    # End-to-end document parsing bypasses layout constraints and category definitions, 
    # yet the HunyuanOCR model produces outputs with more flexible granularities. 
    # For certain mathematical expressions, HunyuanOCR may parse them using a mix of plain texts and embedded LaTeX formulas. 
    # This can lead to mismatches when evaluated with OmniDocBench’s quick-match mechanism. 
    # Therefore, we manually correct cases with incorrect matches (i.e., those with an edit-distance score greater than 0.5). 
    # Below is a piece of logic code used to handle some normalization cases.
    """
    def clean_gt_tail(gt: str):
        pattern = r'(\\quad+|\\qquad+)\s*\{?\(\s*\d+\s*\)\}?\s*$'
        return re.sub(pattern, '', gt).rstrip()
    
    pre_form = clean_gt_tail(pre_form)
    gt_form = gt_form.replace('\[', '').replace('\]', '').replace('，', ',').strip()
    pre_form = pre_form.replace('\[', '').replace('\]', '').replace('，', ',').strip()
    pre_form = pre_form.replace("\\text{ⓐ}", "\\textcircled{a}") \
        .replace("\\text{ⓑ}", "\\textcircled{b}") \
        .replace("\\text{ⓒ}", "\\textcircled{c}") \
        .replace("\\text{ⓓ}", "\\textcircled{d}")
    
    return gt_form, pre_form



def parse_coords(coord_str: str) -> Tuple[float, float]:
    """Parse coordinate string and return (x,y) tuple"""
    try:
        x, y = coord_str.strip('()').split(',')
        return (float(x), float(y))
    except:
        return (0, 0)

def denormalize_coordinates(coord: Tuple[float, float], image_width: int, image_height: int) -> Tuple[int, int]:
    """Denormalize coordinates from [0,1000] to image dimensions""" 
    x, y = coord
    denorm_x = int(x * image_width / 1000)
    denorm_y = int(y * image_height / 1000)
    return (denorm_x, denorm_y)

def process_spotting_response(response: str, image_width: int, image_height: int) -> str:
    """Process spotting task response and denormalize coordinates"""
    try:
        new_response_parts = []
        last_end = 0
        for _text, coord_matches, match in _iter_spotting_items(response):
            new_response_parts.append(response[last_end:match.start()])

            # Parse the two coordinate points 
            if len(coord_matches) == 2:
                start_coord = (float(coord_matches[0][0]), float(coord_matches[0][1]))
                end_coord = (float(coord_matches[1][0]), float(coord_matches[1][1]))
                
                # Denormalize coordinates
                denorm_start = denormalize_coordinates(start_coord, image_width, image_height)
                denorm_end = denormalize_coordinates(end_coord, image_width, image_height)
                
                # Build new coordinate string
                new_coords = f"({denorm_start[0]},{denorm_start[1]}),({denorm_end[0]},{denorm_end[1]})"
                new_response_parts.append(new_coords)
            else:
                new_response_parts.append(match.group(0))
            last_end = match.end()

        new_response_parts.append(response[last_end:])
        return ''.join(new_response_parts)
    
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        return response

def draw_text_detection_boxes(image: Image, response: str) -> Image:
    """Draw text detection boxes on image"""
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Create transparent overlay
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.load_default()
    except IOError:
        font = ImageFont.load_default()
        
    # Extract text and coordinates using bbox delimiters, so text may contain parentheses.
    for text, coord_matches, _match in _iter_spotting_items(response):
        try:
            if len(coord_matches) == 2:
                x1, y1 = int(coord_matches[0][0]), int(coord_matches[0][1])
                x2, y2 = int(coord_matches[1][0]), int(coord_matches[1][1])
                
                # Generate random color
                color = (np.random.randint(0, 200), 
                        np.random.randint(0, 200),
                        np.random.randint(0, 255))
                color_alpha = color + (20,)
                
                # Draw rectangle and overlay
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw_overlay.rectangle([x1, y1, x2, y2], 
                                    fill=color_alpha,
                                    outline=(0, 0, 0, 0))
                
                # Draw text label
                text_x = x1
                text_y = max(0, y1 - 15)
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                draw.rectangle([text_x, text_y, 
                              text_x + text_width, text_y + text_height],
                             fill=(255, 255, 255, 30))
                draw.text((text_x, text_y), text, font=font, fill=color)
                
        except Exception as e:
            print(f"Error drawing box: {str(e)}")
            continue
            
    # Combine image with overlay
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw

def main():
    """Main function to process images and visualize results"""
    # Read JSONL file
    jsonl_path = "/path/to/test_data.jsonl"
    output_dir = "output_visualizations"
    image_root = "/path/to/image_root"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all lines from JSONL
    items = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            items.append(json.loads(line.strip()))
            
    # Randomly select one item
    item = random.choice(items)
    
    # Get image path and response
    image_path = os.path.join(image_root, item["image_name"])
    response = clean_repeated_substrings(item["vllm-infer"])
    
    print(f"Processing image: {item['image_name']}")
    
    # Load and process image
    image = Image.open(image_path)
    image_width, image_height = image.size
    
    # Process response coordinates
    processed_response = process_spotting_response(response, image_width, image_height)
    print("Original response:", response)
    print("Processed response:", processed_response)
    
    # Draw detection boxes
    result_image = draw_text_detection_boxes(image, processed_response)
    
    # Save result using original image name
    output_path = os.path.join(output_dir, item["image_name"])
    result_image.save(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
