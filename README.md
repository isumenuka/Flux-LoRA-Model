# ✨ LoRA Training for "lucataco flux" Style with Custom Faces 🧑‍🎨

This repository provides the code and instructions to train a LoRA (Low-Rank Adaptation) model that applies the **"lucataco flux"** artistic style to *your* custom face images. 🖼️ By using a LoRA, we can efficiently fine-tune a pre-trained Stable Diffusion model to generate images that retain specific facial features while incorporating the desired artistic style. 🚀

This project was inspired by the following:
*   [Pretrained LoRA Model](https://huggingface.co/ezsumm/PradMaz) 🤗
*   [lucataco flux-dev-lora](https://replicate.com/lucataco/flux-dev-lora) 💡

## ✨ Key Features

*   **"lucataco flux" Style:** Trains a LoRA to capture the unique aesthetic of the "lucataco flux" style. 🎨
*   **Custom Face Data:** Enables the model to learn and apply the style to *your* specific face images. 🧑‍🤝‍🧑
*   **Hugging Face Transformers & Diffusers:** Utilizes powerful and accessible libraries for diffusion model training and inference. 📚
*   **Efficient LoRA Training:** Employs the low-rank adaptation technique for fast and resource-friendly fine-tuning. ⚡️

## 🚀 Getting Started

### ✅ Prerequisites

*   **Python 3.8+:** Make sure you have Python 3.8 or a newer version installed. 🐍
*   **CUDA-Enabled GPU:** A CUDA-enabled NVIDIA GPU is highly recommended for training. 💻
*   **Git:** You'll need Git to clone the repository. 🗂️

### 🛠️ Installation

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

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *   **Note:** Make sure `requirements.txt` includes libraries like `accelerate`, `diffusers`, and `transformers`, as well as other needed dependencies. ⚠️

4.  **Organize your custom face data:**
    *   Place your face images inside the `data/custom_faces` directory. 📁
    *   Supported image formats: `.png`, `.jpg`, `.jpeg`. 🖼️

### 🚂 Training the LoRA Model

1.  **Prepare your data:** Ensure your face images are correctly located in `data/custom_faces/`. ✅

2.  **Run the training script:**

    ```bash
    python src/train.py
    ```

    *   **Training Parameters:** You'll likely want to customize the training. The `src/train.py` script uses parameters that can be adjusted to suit your needs.  Commonly used parameters include:
        *   `--pretrained_model_name_or_path`: Base Stable Diffusion model (e.g., `stabilityai/stable-diffusion-2-1`). 💾
        *   `--image_folder`: Path to your face images (`data/custom_faces/`). 🖼️
        *   `--output_dir`:  Directory to save the trained LoRA (e.g., `lora_output/`). 🗂️
        *   `--train_batch_size`:  Batch size during training. ⚙️
        *   `--learning_rate`:  The learning rate for the LoRA fine-tuning. 📈
        *   `--num_train_epochs`: Number of training epochs. 🗓️
        *   `--seed`: A random seed for reproducibility. 🎲
        *   `--mixed_precision`:  Use `fp16` or `bf16` if your system supports it. ✨
    *   **Check `src/train.py` for the full list and how to change them.** 🧐

### 🖼️ Using the Trained LoRA

Once training is done, the LoRA model will be saved in the `lora_output` directory.

1.  **Example Usage:**

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
    prompt = "A portrait of a face in the lucataco flux style"  # Adjust prompt, and add a trigger word if you used one
    image = pipe(prompt, guidance_scale=7.5).images[0]  # Options you can customize
    image.save("images_output/generated_image.png")
    ```

2.  **Customize Generation:** You can adjust `guidance_scale`, `num_inference_steps` in the `pipe()` function to tweak the results. 🎚️

## ⚙️ Customization Options

*   **Data Loading:** Adapt the `src/train.py` data loading logic for special image formats or more complex data setups. 🗄️
*   **Training Configuration:**  Experiment with hyperparameters within the `src/train.py` script to achieve the desired outcome. 🧪
*   **Base Model:** You can specify a different pre-trained Stable Diffusion model by changing `--pretrained_model_name_or_path` in the training script. 💾
*   **Prompt Engineering:** You can get creative with your prompts during image generation and include a custom trigger word if you trained the Lora using one. ✍️

## ✅ Project Goals

*   Provide a method for transferring the "lucataco flux" style onto custom face images. 🖼️➡️🧑‍🎨
*   Make LoRA training accessible using Hugging Face libraries. 🤗
*   Offer a clear and easy-to-follow guide for training your own LoRA. 📖

## 🤝 Contributing

Contributions are welcome! If you find any issues or have ideas for improvements, feel free to open an issue or submit a pull request. 🙋‍♀️🙋‍♂️

## 📝 License

[Specify your license here. Example: MIT License] 📜

## 📧 Contact

isumenuka@gmail.com ✉️
