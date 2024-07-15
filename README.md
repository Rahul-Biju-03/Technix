# Technix

## Problem Statement
Our Problem statement is **“Running GenAI on Intel AI Laptops and Simple LLM Inference on CPU and fine-tuning of LLM Models using Intel® OpenVINO™.”**
The challenge lies in efficiently running Generative AI applications and performing LLM inference on Intel AI Laptops and CPUs, while maintaining high performance without specialized hardware. Additionally, fine-tuning LLM models using Intel® OpenVINO™ for real-time applications requires addressing computational efficiency and resource constraints.

## Objective
This project leverages **Intel® OpenVINO™** to optimize and execute GenAI and LLM inference on Intel AI Laptops' CPUs, minimizing the reliance on GPUs and enabling efficient, high-performance AI deployment in consumer-grade environments. By fine-tuning LLM models with OpenVINO™, we aim to enhance the performance and accessibility of AI applications. Specifically, we have developed a **text generation chatbot** using **TinyLlama/TinyLlama-1.1B-Chat-v1.0** to showcase these capabilities.

## Team Members and Contribution
- **Rahul Biju (Team Leader):** CPU Inference
- **Nandakrishnan A:** Model Optimization and Quantization
- **Nandana S Nair:** Project Report
- **Krishna Sagar P:** Project Report
- **Rahul Zachariah:** User Interface Implementation

## Running locally

**1. Clone the repository.**
```bash
git clone https://github.com/Rahul-Biju-03/Technix.git
```

**2. Move into the project directory.**
```bash
cd Technix
```

**3. Install all the required libraries, by installing the requirements.txt file.**
```bash
pip install -r requirements.txt
```

**4. (Optional) Running it in a virtual environment.**

- Downloading and installing virtualenv.
```bash
pip install virtualenv
```
- Create the virtual environment in Python 3.
```bash
 virtualenv -p path\to\your\python.exe test_env
```
- Activate the test environment.

For Windows:
```bash
test_env\Scripts\Activate
```

For Unix:
```bash
source test_env/bin/activate
```

**5. Converting and Quantizing TinyLlama Model with OpenVINO.**
- This script outlines the steps to convert the TinyLlama model from its original format to ONNX, and subsequently quantize it using OpenVINO for optimized performance.
```bash
python Conversion_and_Optimisation.py
```

**6. Benchmarking Original and Quantized TinyLlama Model with OpenVINO**
- This script benchmarks the performance and memory usage of the original TinyLlama model against the quantized version using OpenVINO, including model size calculations and inference time measurements.
```bash
python CPU_INFERENCE.py
```

**7. TinyLlama Chatbot with Gradio Interface**
- This script sets up a TinyLlama chatbot with a Gradio interface, including preprocessing and postprocessing functions for improved text handling.
```bash
python Chatbot.py
```


## Chatbot Interface
Below are two images illustrating the chatbot interface on a mobile device.
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/64ac9625-8189-45e4-8c78-7fcb24724df5" alt="cove" style="width: 45%; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/4040c593-1f60-4a26-b290-920ad7878967" alt="rep" style="width: 45%;">
</div>



## Demo

https://github.com/user-attachments/assets/4c201111-63d4-4fcd-a984-eec6f2ff6dad






