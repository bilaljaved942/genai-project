# StyleGAN2 Latent Space Face Manipulation

This repository contains the code for a Generative AI project focused on **Latent Space Manipulation using StyleGAN2**. We use a pretrained ResNet18 encoder to map real face photos into the StyleGAN2 `W+` latent space, allowing us to edit facial attributes (like hair color, age, and smile) in real-time.

## Project Structure

*   `app.py` - A Gradio web application that provides an interactive UI with sliders for real-time attribute manipulation.
*   `compute_directions.py` - Script to calculate attribute direction vectors (`.pth`) by finding the mean difference in the `W+` space between positive and negative examples (using the CelebA dataset).
*   `test_reconstruction.py` - A CLI testing script to verify the encoder's reconstruction quality on input images.
*   `training.ipynb` - The original Jupyter Notebook containing the `FaceEncoder` training pipeline (PyTorch), W-space regularization, and early-stopping logic.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd genai-project
    ```

2.  **Create a Python environment:**
    ```bash
    conda create -n stylegan_env python=3.11 -y
    conda activate stylegan_env
    ```

3.  **Install PyTorch and dependencies:**
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install gradio pandas pillow matplotlib
    ```

4.  **Download required models:**
    *   Clone the official `stylegan2-ada-pytorch` repository into this folder:
        ```bash
        git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
        ```
    *   Download the pretrained FFHQ generator weights (`ffhq.pkl`, ~380MB) and place it in the root folder.
    *   Place the trained `encoder_best.pth` file in the root folder.

## How to Run the Web App

Once the models are in place and dependencies are installed, you can launch the Gradio interface:

```bash
python app.py
```

This will start a local web server. Open the provided `http://localhost:7860` link in your browser to interact with the model!

## How it Works

1.  **Encoder:** A custom ResNet18-based architecture maps a $256 \times 256$ RGB image into an $18 \times 512$ dimensional tensor (the `W+` space).
2.  **Direction Vectors:** By finding the difference between average latent codes of faces *with* an attribute (e.g., Blond) and *without* the attribute (e.g., Dark hair), we get a linear direction vector.
3.  **Manipulation:** The UI sliders add or subtract these direction vectors from the input image's latent code.
4.  **Generation:** The modified `W+` latent code is passed to the StyleGAN2 generator to synthesize the edited image.

## Excluded Files

Note: Large model weights (`.pth`, `.pkl`) and datasets (CelebA) are excluded from this repository via `.gitignore` to keep it lightweight.
