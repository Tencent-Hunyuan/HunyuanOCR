<div align="center">

[中文阅读](./README_zh.md)

</div>

<div align="center">

# HunyuanOCR

</div>

<p align="center">
 <img src="./assets/hyocr-head-img.png" width="80%"/> <br>
</p>


<p align="center">

<a href="http://29.210.129.176:8080/"><b>🎯 Demo</b></a> |

<a href="https://huggingface.co/tencent/HunyuanOCR"><b>📥 Model Download</b></a> |

<a href="./Hunyuan_OCR_Technical_Report.pdf"><b>📄 Technical Report</b></a>

</p>

## 🔥 News
- **[2025/11/25]** 📝 Inference code and model weights publicly available.

## 📖 Introduction
**HunyuanOCR** stands as a leading end-to-end OCR expert VLM powered by Hunyuan's native multimodal architecture. With a remarkably lightweight 1B parameter design, it has achieved multiple state-of-the-art benchmarks across the industry. The model demonstrates mastery in **complex multilingual document parsing** while excelling in practical applications including **text spotting, open-field information extraction, video subtitle extraction, and photo translation**.

Built on Tencent's Hunyuan technology, this versatile model delivers exceptional performance through end-to-end architecture design and single-pass inference. It significantly simplifies deployment while maintaining competitive performance against both established cascade systems and commercial APIs.

## ✨ Key Features

- 💪 **Efficient Lightweight Architecture**: Built on Hunyuan's native multimodal architecture and training strategy, achieving SOTA performance with only 1B parameters, significantly reducing deployment costs.

- 📑 **Comprehensive OCR Capabilities**: A single model covering classic OCR tasks including text detection and recognition, complex document parsing, open-field information extraction and video subtitle extraction, while supporting end-to-end photo translation and document QA.

- 🚀 **Ultimate Usability**: Deeply embraces the "end-to-end" philosophy of large models - achieving SOTA results with single instruction and single inference, offering greater efficiency and convenience compared to industry cascade solutions.

- 🌏 **Extensive Language Support**: Robust support for over 100 languages, excelling in both single-language and mixed-language scenarios across various document types.

<div align="left">
  <img src="./assets/hyocr-pipeline.png" alt="HunyuanOCR framework" width="80%">
</div>




## 🛠️ Dependencies and Installation

### System Requirements
- 🖥️ Operating System: Linux
- 🐍 Python: 3.12+ (recommended and tested)
- ⚡ CUDA: 12.8
- 🔥 PyTorch: 2.7.1
- 🎮 GPU: NVIDIA GPU with CUDA support
- 🧠 GPU Memory: 80GB
- 💾 Disk Space: 6GB

### Installation
```bash
pip install https://mirrors.tencent.com/repository/generic/transformers/transformers-4.56.0.post2-py3-none-any.whl
pip install opencv-python-headless
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```



## 🚀 Quick Start with Transformers

### Model Inference

```python
from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration
from PIL import Image
import torch

model_name_or_path = "tencent/HunyuanOCR"
processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
img_path = "path/to/your/image.jpg"
image_inputs = Image.open(img_path)
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": (
                "Extract all information from the main body of the document image "
                "and represent it in markdown format, ignoring headers and footers. "
                "Tables should be expressed in HTML format, formulas in the document "
                "should be represented using LaTeX format, and the parsing should be "
                "organized according to the reading order."
            )},
        ],
    }
]
messages = [messages1]
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
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
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)
```

### Alternatively, you can also use the provided demo script as follow:
```shell
cd Hunyuan-OCR-master/Hunyuan-OCR-hf && python run_hy_ocr.py
```

## 💬 Application-oriented Prompts

| Task | English | Chinese |
|------|---------|---------|
| **Spotting** | Detect and recognize text in the image, and output the text coordinates in a formatted manner. | 检测并识别图片中的文字，将文本坐标格式化输出。 |
| **Parsing** | • Identify the formula in the image and represent it using LaTeX format.<br><br>• Parse the table in the image into HTML.<br><br>• Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.<br><br>• Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order. | • 识别图片中的公式，用 LaTeX 格式表示。<br><br>• 把图中的表格解析为 HTML。<br><br>• 解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示。<br><br>• 提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。 |
| **Information Extraction** | • Output the value of Key.<br><br>• Extract the content of the fields: ['key1','key2', ...] from the image and return it in JSON format.<br><br>• Extract the subtitles from the image. | • 输出 Key 的值。<br><br>• 提取图片中的: ['key1','key2', ...] 的字段内容，并按照 JSON 格式返回。<br><br>• 提取图片中的字幕。 |
| **Translation** | First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format. | 先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。 |


## 📊 Evaluation

> **Note**: Evaluation metrics for competing methods are taken from official reports when available; otherwise, they are reproduced using competitor models or interfaces with the recommended standard instructions.

> **Note**: The HunyuanOCR evaluation metrics are derived using the TensorRT framework, which may slightly differ from the inference methods using Transformers or vLLM.

### Text Spotting Performance on In-house Benchmark

| Model Type | Methods | Overall | Art | Doc | Game | Hand | Ads | Receipt | Screen | Scene | Video |
|------------|---------|---------|-----|-----|------|------|-----|----------|---------|--------|--------|
| **Traditional methods** | PaddleOCR | 53.38 | 32.83 | 70.23 | 51.59 | 56.39 | 57.38 | 50.59 | 63.38 | 44.68 | 53.35 |
| **Traditional methods** | BaiduOCR | 61.9 | 38.5 | **78.95** | 59.24 | 59.06 | 66.7 | **63.66** | 68.18 | 55.53 | 67.38 |
| **General VLM** | Qwen3VL-2B-Instruct | 29.68 | 29.43 | 19.37 | 20.85 | 50.57 | 35.14 | 24.42 | 12.13 | 34.90 | 40.1 |
| **General VLM** | Qwen3VL-235B-Instruct | 53.62 | 46.15 | 43.78 | 48.00 | 68.90 | 64.01 | 47.53 | 45.91 | 54.56 | 63.79 |
| **General VLM** | Seed-1.6-Vision | 59.23 | 45.36 | 55.04 | 59.68 | 67.46 | 65.99 | 55.68 | 59.85 | 53.66 | 70.33 |
| **OCR-Specific VLM** | HunyuanOCR | **70.92** | **56.76** | 73.63 | **73.54** | **77.10** | **75.34** | 63.51 | **76.58** | **64.56** | **77.31** |

> **Summary**: HunyuanOCR achieves the best overall performance (70.92%) across different scenarios, significantly outperforming both traditional OCR methods and general VLMs.

### Document Parsing Performance on OmniDocBench and Multilingual In-house Benchmark (Edit Distance)

| Model Type | Method | Size | OmniDocBench | | | | Wild-OmniDocBench | | | | DocML |
|:-----------|:-------|:-----|:---------|:---------|:----------|:--------|:----------|:---------|:----------|:---------|:--------|
| | | | overall | text | formula | table | overall | text | formula | table | |
| **General VLMs** | Gemni-2.5-pro | - | 88.03 | 0.075 | 85.92 | 85.71 | - | - | - | - | 82.64 |
| **General VLMs** | Qwen3-VL-235B | 235B | 89.15 | 0.069 | 88.14 | 86.21 | 79.69 | 0.09 | 80.67 | 68.31 | 81.40 |
| **Specialized VLMs (Modular)** | MonkeyOCR-pro-3B | 3B | 88.85 | 0.075 | 87.5 | 86.78 | 70.00 | 0.211 | 63.27 | 67.83 | 56.50 |
| **Specialized VLMs (Modular)** | MinerU2.5 | 1.2B | 90.67 | 0.047 | 88.46 | 88.22 | 70.91 | 0.218 | 64.37 | 70.15 | 52.05 |
| **Specialized VLMs (Modular)** | PaddleOCR-VL | 0.9B | 91.93 | 0.039 | 88.67 | 91.01 | 72.19 | 0.232 | 65.54 | 74.24 | 57.42 |
| **Specialized VLMs (End2End)** | Mistral-OCR | - | 78.83 | 0.164 | 82.84 | 70.03 | - | - | - | - | 64.71 |
| **Specialized VLMs (End2End)** | Deepseek-OCR | 3B | 87.01 | 0.073 | 83.37 | 84.97 | 74.23 | 0.178 | 70.07 | 70.41 | 57.22 |
| **Specialized VLMs (End2End)** | dots.ocr | 3B | 88.41 | 0.048 | 83.22 | 86.78 | 78.01 | 0.121 | 74.23 | 71.89 | 77.50 |
| **Specialized VLMs (End2End)** | **HunyuanOCR** | 1B | **94.10** | 0.042 | **94.73** | **91.81** | **85.21** | **0.081** | **82.09** | **81.64** | **91.03** |


> **Summary**: HunyuanOCR demonstrates superior performance in multilingual document parsing, achieving the lowest edit distances across most categories.

### Information Extraction (in-house Benchmark) and VQA Performance (OCRBench)

| Model | Cards | Receipts | Video Subtitles | OCRBench |
|:------|:------|:---------|:----------------|:----------|
| DeepSeek-OCR | 10.04 | 40.54 | 5.41 | 430 |
| PP-ChatOCR | 57.02 | 50.26 | 3.1 | - |
| Qwen3-VL-2B-Instruct | 67.62 | 64.62 | 3.75 | 858 |
| Seed-1.6-Vision | 70.12 | 67.5 | 60.45 | 881 |
| Qwen3-VL-235B-A22B-Instruct | 75.59 | 78.4 | 50.74 | **920** |
| Gemini-2.5-Pro | 80.59 | 80.66 | 53.65 | 872 |
| **HunyuanOCR (∼1B)** | **92.29** | **92.53** | **92.87** | 860 |


> **Summary**: HunyuanOCR significantly outperforms larger models in cards/receipts processing and video subtitle extraction, while maintaining competitive performance on OCRBench.

### Text Image Transation (in-house Benchmark) Performance

| Method | Size | Other2En | Other2Zh | DoTA (en2zh) |
|--------|------|-----------|-----------|--------------|
| Gemini-2.5-Flash | - | 79.26 | 80.06 | 85.60 |
| Qwen3-VL-235B-Instruct | 235B | 73.67 | 77.20 | 80.01 |
| Qwen3-VL-8B-Instruct | 4B | 75.09 | 75.63 | 79.86 |
| Qwen3-VL-4B-Instruct | 4B | 70.38 | 70.29 | 78.45 |
| Qwen3-VL-2B-Instruct | 2B | 66.30 | 66.77 | 73.49 |
| **HunyuanOCR** | **1B** | 73.38 | 73.62 | 83.48 |

> **Summary**: HunyuanOCR using only 1B of parameters, achieved comparable results to Gemini-2.5-Flash and Qwen3-VL-235B in photo translation tasks.

## 💡 Visualizations
<details>
<summary><u style="color: #2E64FE;">Click here to view detailed results.</u></summary>


### Text Spotting

Our model aims to output the text content and corresponding coordinate information of all text appearing in a text image at the line level. It performs exceptionally well in scenarios such as documents, artistic fonts, street views, handwriting, advertisements, invoices, screenshots, games, and videos.

<p align="left">
 <img src="./assets/spotting1_cropped.png" width="40%"/> <br>
 <img src="./assets/vis_document_23.jpg" width="40%"/> <br>
</p>


### Complex Document Processing

Digitizing scanned or photographed images of multilingual documents involves, specifically, organizing the text content within the images according to reading order, using LaTeX format for formulas, and expressing complex tables in HTML format.

<p align="left">
 <img src="./assets/vis_parsing_fig.png" width="40%"/> <br>
  <img src="./assets/show_res_parsing_fig.png" width="40%"/> <br>
  <img src="./assets/vis_parsing_table.png" width="40%"/> <br>
  <img src="./assets/vis_parsing_table_2.png" width="40%"/> <br>
  <img src="./assets/parsing_rgsj.png" width="40%"/> <br>
  <img src="./assets/parsing_rgsjz_2.png" width="40%"/> <br>
  <img src="./assets/qikai1.png" width="40%"/> <br>
  <img src="./assets/guwan1.png" width="40%"/> <br>
  <img src="./assets/parsing_chart1.png" width="40%"/> <br>
  <img src="./assets/vis_parsing_chart1.png" width="40%"/> <br>
  <img src="./assets/vis_parsing_chart2.png" width="40%"/> <br>
  <img src="./assets/vis_parsing_chart3.png" width="40%"/> <br>
</p>



### Open-field Information Extraction

For common cards and tickets, fields of interest (such as name/address/company) are parsed using standard JSON format.

<p align="left">
 <img src="./assets/vis_ie_1.png" width="40%"/> <br>
</p>

<p align="left">
 <img src="./assets/ie_parallel.jpg" width="25%"/> <br>
</p>

**Prompt:**
Extract the content of the fields: ['单价', '上车时间', '发票号码', '省前缀', '总金额', '发票代码', '下车时间', '里程数'] from the image and return it in JSON format.

**Response:**
```json
{
    "单价": "3.00",
    "上车时间": "09:01",
    "发票号码": "42609332",
    "省前缀": "陕",
    "总金额": "￥77.10元",
    "发票代码": "161002018100",
    "下车时间": "09:51",
    "里程数": "26.1km"
}
```

### Video Subtitle Extraction

Our model is capable of automatically extracting subtitles from videos, including bilingual ones.

<p align="left">
 <img src="./assets/vis_subtitle1.png" width="40%"/> <br>
 <img src="./assets/vis_subtitle2.png" width="40%"/> <br>
 <img src="./assets/vis_subtitle3.png" width="37.5%"/> <br>
</p>



### Image Text Translation

Our model is able to translate images of minor languages ​​taken into Chinese or English text format end-to-end. Currently, it mainly supports 14 frequently used minor languages ​​(specifically including: German, Spanish, Turkish, Italian, Russian, French, Portuguese, Arabic, Thai, Vietnamese, Indonesian, Malay, Japanese, and Korean) into Chinese/English, as well as Chinese-English translation function (it won the small model track championship in the ICDAR2025 document end-to-end translation competition).

<p align="left">
 <img src="./assets/translation2.png" width="40%"/> <br>
</p>

</details>


## 📚 Citation
@misc{hunyuanocr2025,
    title={HunyuanOCR: Advanced OCR Engine for Document Understanding},
    author={Tencent Hunyuan Team},
    year={2025},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/Tencent/HunyuanOCR}}
}

## 🙏 Acknowledgements
Thanks to all contributors who helped build HunyuanOCR
Special thanks to the Tencent Hunyuan Team
We appreciate the support from the open-source community
