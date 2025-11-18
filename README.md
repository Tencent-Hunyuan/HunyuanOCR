<div align="left">

[中文](./README_zh.md) | [English](./README.md)

</div>

# HunyuanOCR

<p align="left">
 <img src="./assets/hyocr-head-img.png" width="800"/> <br>
</p>

<div align="center">

📑 Paper & Model Weights are coming soon!


[Demo](#demo) | [Install](#dependencies-and-installation) | [Quick Start](#quick-start-with-transformers) | [Prompt Templates](#application-oriented-prompts) | [Documentation](#citation)

</div>

## 🔥 News
- **[2025/11/20]** 📝 Inference code and model weights publicly available.

## 📖 Introduction
**HunyuanOCR** stands as a leading end-to-end OCR expert VLM powered by Hunyuan's native multimodal architecture. With a remarkably lightweight 1B parameter design, it has achieved multiple state-of-the-art benchmarks across the industry. The model demonstrates mastery in **complex document parsing** while excelling in practical applications including **multilingual text recognition, open-field information extraction, video subtitle extraction, and photo translation**.

Built on Tencent's Hunyuan technology, this versatile model delivers exceptional performance through end-to-end architecture design and single-pass inference. It significantly simplifies deployment while maintaining competitive performance against both established cascade systems and commercial APIs.

## ✨ Key Features

- 💪 **Efficient Lightweight Architecture**: Built on Hunyuan's native multimodal architecture and training strategy, achieving SOTA performance with only 1B parameters, significantly reducing deployment costs.

- 📑 **Comprehensive OCR Capabilities**: A single model covering classic OCR tasks including text detection and recognition, complex document parsing, open-field information extraction and video subtitle extraction, while supporting end-to-end photo translation and document QA.

- 🚀 **Ultimate Usability**: Deeply embraces the "end-to-end" philosophy of large models - achieving SOTA results with single instruction and single inference, offering greater efficiency and convenience compared to industry cascade solutions.

- 🌏 **Extensive Language Support**: Robust support for over 100 languages, excelling in both single-language and mixed-language scenarios across various document types.

<div align="left">
  <img src="./assets/hyocr-pipeline.png" alt="HunyuanOCR framework" width="50%">
</div>




## 🛠️ Dependencies and Installation

### System Requirements
- 🖥️ Operating System: Linux
- 🐍 Python: 3.12+ (recommended and tested)
- ⚡ CUDA: 12.8
- 🔥 PyTorch: 2.7.1
- 🎮 GPU: NVIDIA GPU with CUDA support
- 🧠 GPU Memory: ≥3×80GB (4×80GB recommended for better performance)
- 💾 Disk Space: 170GB for model weights

### Installation
```bash
pip install https://mirrors.tencent.com/repository/generic/transformers/transformers-4.56.0.post2-py3-none-any.whl
pip install opencv-python-headless
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```



## 🚀 Quick Start with Transformers

### Model Inference

```python
from transformers import AutoModel, AutoProcessor
import torch
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load model and processor
model_name = "Tencent/HunyuanOCR"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda()

# Single image inference
image_path = "path/to/your/image.jpg" 
prompt = "<image>\nExtract text from the image."
output_path = "path/to/output"

result = model.infer(
    processor,
    prompt=prompt,
    image_file=image_path,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True
)
print("Inference result:", result)
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

### Text Spotting Performance on In-house Benchmark

| Model Type | Methods | Overall | Art | Doc | Game | Hand | Ads | Receipt | Screen | Scene | Video |
|------------|---------|---------|-----|-----|------|------|-----|----------|---------|--------|--------|
| **Traditional methods** | PaddleOCR | 53.38 | 32.83 | 70.23 | 51.59 | 56.39 | 57.38 | 50.59 | 63.38 | 44.68 | 53.35 |
| | BaiduOCR | 61.9 | 38.5 | **78.95** | 59.24 | 59.06 | 66.7 | **63.66** | 68.18 | 55.53 | 67.38 |
| **General VLM** | Qwen3VL-2B-Instruct | 29.68 | 29.43 | 19.37 | 20.85 | 50.57 | 35.14 | 24.42 | 12.13 | 34.90 | 40.1 |
| | Qwen3VL-235B-Instruct | 53.62 | 46.15 | 43.78 | 48.00 | 68.90 | 64.01 | 47.53 | 45.91 | 54.56 | 63.79 |
| | Seed1.6-VL-Instruct | 59.23 | 45.36 | 55.04 | 59.68 | 67.46 | 65.99 | 55.68 | 59.85 | 53.66 | 70.33 |
| **OCR-Specific VLM** | HunyuanOCR | **70.92** | **56.76** | 73.63 | **73.54** | **77.10** | **75.34** | 63.51 | **76.58** | **64.56** | **77.31** |

> **Summary**: HunyuanOCR achieves the best overall performance (70.92%) across different scenarios, significantly outperforming both traditional OCR methods and general VLMs.

### Document Parsing Performance on OmniDocBench (Edit Distance)

| Model | English |  |  |  | Chinese |  |  |  |
|-------|---------|---------|----------|--------|----------|---------|----------|---------|
|  | overall | text | formula | table | overall | text | formula | table |
| Dolphin | 0.356 | 0.352 | 0.465 | 0.258 | 0.44 | 0.44 | 0.604 | 0.367 |
| SmolDocling | 0.493 | 0.262 | 0.753 | 0.729 | 0.816 | 0.838 | 0.997 | 0.907 |
| dots.ocr-3B | 0.182 | 0.137 | 0.320 | 0.166 | 0.261 | 0.229 | 0.468 | 0.160 |
| HunyuanOCR | **0.123** | 0.049 | **0.242** | 0.147 | **0.157** | 0.087 | **0.377** | **0.08** |

> **Summary**: HunyuanOCR demonstrates superior performance in both English and Chinese document parsing, achieving the lowest edit distances across most categories.

### Information Extraction (in-house Benchmark) and VQA Performance (OCRBench)

| Model | Cards & Receipts | Video Subtitles | OCRBench |
|-------|------------------|-----------------|-----------|
| DeepSeek-OCR | 25.29 | 5.41 | 430 |
| PP-ChatOCR | 53.64 | 3.1 | - |
| Qwen3VL-2B-Instruct | 66.12 | 3.75 | 858 |
| Seed1.6-VL | 68.81 | 60.45 | 881 |
| Qwen3VL-235B-A22B-Instruct | 77.0 | 50.74 | **920** |
| Gemini-2.5-Pro | 80.63 | 53.65 | 872 |
| **HunyuanOCR (∼1B)** | **92.41** | **92.87** | 860 |

> **Summary**: HunyuanOCR significantly outperforms larger models in cards/receipts processing and video subtitle extraction, while maintaining competitive performance on OCRBench.

### Text Image Transation (in-house Benchmark) Performance

| Method | Size | Other2En | Other2Zh | DoTA (en2zh) |
|--------|------|-----------|-----------|--------------|
| Gemini-2.5-Flash | - | 79.26 | 80.06 | - |
| Qwen3-VL-235B | 235B | 73.67 | 77.20 | - |
| Qwen3-VL-4B | 4B | 70.38 | 70.29 | - |
| Qwen3-VL-2B | 2B | 66.30 | 66.77 | - |
| **HunyuanOCR** | **1B** | 73.38 | 73.62 | - |

> **总结**: HunyuanOCR using only 1B of parameters, achieved comparable results to Gemini-2.5-Flash and Qwen3-VL-235B in photo translation tasks.

## 💡 Visualizations
<details>
<summary>Click here to view detailed results.</summary>

### Spotting


<p align="left">
 <img src="./assets/spotting1_cropped.png" width="20%"/> <br>
 <img src="./assets/vis_document_23.jpg" width="20%"/> <br>
</p>


### Document Processing
<p align="left">
 <img src="./assets/vis-parsing.png" width="20%"/> <br>
</p>


### Information Extraction


<p align="left">
 <img src="./assets/ie_parallel.jpg" width="8%"/> <br>
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

<p align="left">
 <img src="./assets/zimu2.jpg" width="20%"/> <br>
</p>

**Prompt:**
Extract the subtitles from the image.

**Response:**
对不起 我是诺曼·斯佩尔曼\nSorry, I'm Norm. Spellman.



### Translation
<p align="left">
 <img src="./assets/vis-translation.png" width="20%"/> <br>
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


## 📄 License
This project is licensed under the Apache 2.0 License.
