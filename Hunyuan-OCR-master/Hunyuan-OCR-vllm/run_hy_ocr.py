import json
import base64
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List
import mimetypes

def encode_image(image_path: str) -> str:
    """
    Encode image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_chat_messages(image_path: str, prompt: str) -> List[Dict]:
    """
    Create chat messages with image and prompt.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt for the model
        
    Returns:
        List of message dictionaries
    """
    # Detect MIME type (jpg/png/webp/etc)
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/jpeg"  # fallback

    return [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": f"data:{mime};base64,{encode_image(image_path)}"
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

def process_single_item(client: OpenAI, data: Dict) -> Dict:
    """
    Process a single data item through the VLLM API.
    
    Args:
        client: OpenAI client instance
        data: Input data dictionary
        
    Returns:
        Updated data dictionary with model response
    """
    # Extract image path and prompt
    img_path = data['image_path']
    prompt = data['question']
    
    # Create chat messages
    messages = create_chat_messages(img_path, prompt)
    
    # Get model response
    response = client.chat.completions.create(
        model="tencent/HunyuanOCR",
        messages=messages,
        temperature=0.0,
        top_p=0.95,
        seed=1234,
        stream=False,
        extra_body={
            "top_k": 1,
            "repetition_penalty": 1.0
        }
    )
    
    # Update data with model response
    data["vllm_answer"] = response.choices[0].message.content
    return data

def main():
    """Main function to process the JSONL file through VLLM API"""
    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=3600
    )
    
    # Define input/output paths
    input_path = 'ominidoc_bench.jsonl'
    output_path = "infer_result_ominidoc_bench.jsonl"
    
    # Process data
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        # Iterate through input file
        for line in tqdm(fin, desc="Processing documents"):
            if not line.strip():
                continue
                
            try:
                # Load and process data
                data = json.loads(line)
                processed_data = process_single_item(client, data)
                
                # Write results
                fout.write(json.dumps(processed_data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
    
    print(f"Processing completed. Results saved to: {output_path}")

if __name__ == "__main__":
    main()
