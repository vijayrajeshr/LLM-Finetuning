# Use an official Python runtime with PyTorch and CUDA pre-installed to save build time
# Alternatively, python:3.10-slim can be used, but downloading PyTorch from scratch takes longer.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code, including the LoRA adapter folder
COPY . .

# IMPORTANT: test_fine_tuning.py takes user input from the terminal via 'input()'.
# To allow interactive inputs when running the docker container, we'll need the '-it' flag.
CMD ["python", "test_fine_tuning.py"]
