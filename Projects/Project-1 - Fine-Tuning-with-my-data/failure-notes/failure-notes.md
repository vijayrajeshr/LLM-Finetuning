Project Overview
Status: Development-Phase

Platform: Optimized for Google Colab to utilize the free Tesla T4 GPU with 15GB VRAM

Base Model: Llama 3.2 (3B Instruct) loaded in 4-bit to conserve VRAM

Identity and Professional Metrics
The model is fine-tuned with specific synthesized data:

Identity: Vijay Rajesh R, a B.Tech AIML student at VIT Bhopal (2023-2027)

Certifications: AWS Certified Cloud Practitioner (Feb 2026) and Oracle Certified Gen-AI Professional

Impact: Increased trend insight accuracy by 25% at MedTourEasy and achieved 85%+ accuracy in 6+ ML models at Edu Versity

Philosophy: An agile, highly-productive "Supertasker" turning research into prototypes

Project K: An open-source Windows-based application initiative

Understanding LoRA and Adapters
Low-Rank Adaptation (LoRA) allows for parameter-efficient fine-tuning:

Instead of changing billions of parameters, the process targets specific attention layers such as q_proj and v_proj

Two small matrices with a Rank of r=16 are created alongside the main model

Only these small matrices are updated during training, while the base knowledge of Llama 3.2 remains frozen

The "Delta" represents the new weights holding specific persona and metric data

The Meaning of Checkpoint-60
A checkpoint is a serialized snapshot of the model's weights and biases at a specific moment in training

The number 60 corresponds to the final training step, as the process was configured with max_steps = 60

Checkpoint-60 represents the finalized state of intelligence after processing all career data and metrics

File Organization and Purpose
Essential Adapter Files
These are required to run, share, or plug the "Vijay-LLM" persona into the base model:

adapter_model.safetensors: The most critical file containing the learned weights and professional history

adapter_config.json: The instruction manual or map that tells the base model how to integrate the custom weights

tokenizer.json / tokenizer_config.json: The dictionary used to translate technical words into numerical format

chat_template.jinja: Formatting instructions for managing User vs. Assistant conversation formats

Non-Essential Training Artifacts
These are used by the computer during the training process but are not needed for the final AI to function:

optimizer.pt: Mathematical notes regarding GPU momentum

scheduler.pt: Records the learning rate timing

rng_state.pth / scaler.pt: Randomness and precision settings

trainer_state.json / training_args.bin: Logs of the training settings

Model Export
The final step involves preparing the model for inference and exporting it to GGUF format:

This process merges the VIT Bhopal status and MedTourEasy metrics into a single file

The quantization method "q4_k_m" is used to optimize the file for use