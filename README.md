# LoRA Training for the "lucataco flux" Aesthetic with Custom Faces

This repository contains the code and resources to train a LoRA model that captures the artistic style of "lucataco flux" and applies it to custom faces. This allows you to generate images with the desired aesthetic while preserving specific facial features.

## Overview

This project utilizes:

*   **Hugging Face Transformers:**  For accessing pre-trained Stable Diffusion models and training LoRAs.
*   **Diffusers:** Hugging Face's library for diffusion models, used for model loading and LoRA application.
*   **Custom Face Dataset:** Allows the model to learn to apply the lucataco flux style to *your* specific faces.
*   **LoRA (Low-Rank Adaptation):** An efficient fine-tuning technique that allows adapting the diffusion model to the desired style.
*   **Python 3.8+**

## Repository Structure
Use code with caution.
Markdown
.
├── data/
│ └── custom_faces/ # Images of custom faces
│ ├── face1.png
│ ├── face2.jpg
│ └── ...
├── src/
│ ├── train.py # Main training script
│ ├── utils.py # Utility functions
│ └── ...
├── lora_output/ # Output LoRA model files
├── images_output/ # Generated images using the trained LoRA
├── requirements.txt # List of required Python packages
└── README.md # This file

## Getting Started

### Prerequisites

1.  **Python 3.8+:** Make sure you have Python 3.8 or a later version installed.
2.  **CUDA-enabled GPU:** Training diffusion models requires a capable GPU.
3.  **Git:** To clone this repository.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/your_repo_name.git
    cd your_repo_name
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```
    **NOTE:** `requirements.txt` should include `accelerate`, `diffusers`, `transformers` and other required packages.

4.  **Organize custom face dataset:** Place the image files of your custom faces in the `data/custom_faces` directory.

### Training the LoRA

1.  **Prepare the training data:**
    *   Ensure your face images in `data/custom_faces` are correctly named (e.g., `face1.png`, `face2.jpg`, etc.).
    *   Consider using a script to create caption files if needed.

2.  **Run the training script:**

    ```bash
    python src/train.py
    ```
  *  **Parameters:**  You might need to customize training script parameters to suit your needs. Parameters are included in `src/train.py` file, but some of them are:
    * `--pretrained_model_name_or_path`: base diffusion model for training (e.g. stabilityai/stable-diffusion-2-1).
    * `--image_folder`: Path to `data/custom_faces/`.
    * `--output_dir`: Path for saving the trained Lora.
    * `--train_batch_size`: Training batch size.
    * `--learning_rate`: Learning rate for LoRA training.
    * `--num_train_epochs`: Number of epochs to train.
    * `--seed`: Random seed.
    * `-- mixed_precision`: If your system supports it, it should be used (fp16 or bf16).

    *   Check the script `src/train.py` for full usage and customization of parameters.

### Using the Trained LoRA

Once the training is complete, your LoRA model will be saved in the `lora_output` directory. You can then load it using diffusers and apply it to Stable Diffusion pipelines to generate images.

1.  **Example usage with Diffusers:**

    ```python
    from diffusers import DiffusionPipeline, StableDiffusionPipeline
    import torch

    # Load the base Stable Diffusion model
    base_model_path = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16).to("cuda")

    # Load the trained LoRA weights
    lora_path = "lora_output/your_lora_model_name.safetensors"  # Replace with your LoRA file name
    pipe.load_lora_weights(lora_path)

    # Generate an image
    prompt = "A portrait of a face in the lucataco flux style"  # Adjust prompt as needed. Add training trigger word if you have one.
    image = pipe(prompt, guidance_scale=7.5).images[0]  # Optional parameters.
    image.save("images_output/generated_image.png")  # Replace with the desired output path
    ```
    
2.  **Parameters:**  You can play around with `guidance_scale` and `num_inference_steps` for generating different results.

## Customization

*   **Dataset:**  Feel free to modify the data loading logic in `src/train.py` to suit your specific data format.
*   **Training Parameters:** Adjust training parameters in `src/train.py` (like learning rate, batch size, etc.) to experiment with the result.
*   **Base Model:** You can change the base Stable Diffusion model using the `--pretrained_model_name_or_path` parameter.
*   **Prompt Engineering:** Experiment with different prompts to generate images with the right content and style.
*   **More Complex Usage:**  Add a training trigger word for the LoRA if needed, and use it in your prompt.

## Project Goals

*   **Learn LoRA Training:** Understand the process of training LoRAs using Hugging Face Transformers.
*   **Style Transfer:** Achieve style transfer from the "lucataco flux" aesthetic onto custom faces.
*   **Facial Preservation:** Generate images that retain the facial characteristics of your specific dataset.
*   **Share the Process:** Share the process of training LoRAs and custom datasets, enabling other enthusiasts to replicate your results.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports or feature requests.

## License

[Specify your license here (e.g., MIT License, Apache License 2.0)]

## Contact

[Your contact information or email address]

---

This README.md provides a solid starting point for your project. Be sure to adapt it to your specific needs and add details where necessary. Good luck!
