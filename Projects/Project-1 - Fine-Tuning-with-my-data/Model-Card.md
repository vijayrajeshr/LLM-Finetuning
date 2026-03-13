---
base_model: unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
- vijay-rajesh-r
---

# Model Card for Vijay Rajesh R Profile Assistant

[cite_start]This model is a lightweight LoRA adapter fine-tuned on the professional profile of Vijay Rajesh R to act as a specialized personal assistant.

## Model Details

### Model Description

[cite_start]This model is fine-tuned to answer questions regarding the education, certifications, and technical skills of Vijay Rajesh R, including his background in AIML and his certifications from AWS and Oracle.

- [cite_start]**Developed by:** Vijay Rajesh R 
- [cite_start]**Model type:** PEFT (LoRA) Adapter 
- [cite_start]**Language(s) (NLP):** English 
- [cite_start]**Finetuned from model:** unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit 

### Model Sources

- [cite_start]**Repository:** [https://github.com/vijayrajeshr](https://github.com/vijayrajeshr) 

## Uses

### Direct Use

[cite_start]The model is intended for personal portfolio use or as a chatbot to provide professional information about Vijay Rajesh R.

### Out-of-Scope Use

[cite_start]The model should not be used for critical decision-making or to provide information outside the scope of the provided professional dataset.

## Bias, Risks, and Limitations

[cite_start]The model's knowledge is limited to the provided professional summary and may not reflect real-time updates to Vijay's career or personal life.

### Recommendations

[cite_start]Users should verify specific credentials (like AWS or Oracle IDs) against official issuers if using the model for recruitment purposes.

## How to Get Started with the Model

To run this model locally with Ollama, use the following `Modelfile`:

```
FROM llama3.2:latest
ADAPTER .
SYSTEM "You are an AI assistant representing Vijay Rajesh R."

```

Training DetailsTraining DataThe model was trained on a custom .jsonl dataset containing personal information, education (VIT Bhopal), certifications, and project details.Training ProcedureTraining HyperparametersTraining regime: 

LoRA fine-tuning with 4-bit quantization.Rank (r): 16 Learning Rate: 2e-4 

Steps: 60 Technical SpecificationsCompute InfrastructureHardwareHardware 

Type: T4 GPU (Google Colab) SoftwareLibrary: Unsloth Framework: PEFT 0.18.1 

Model Card 
ContactEmail: vijayrajeshr@gmail.com 

