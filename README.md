# Robustness of AI-Based Medical Imaging Under Real-World Image Compression: A MONAI Study on WhatsApp-Shared Radiographs

This repository contains the source code and supporting resources for the article "Robustness of AI-Based Medical Imaging Under Real-World Image Compression: A MONAI Study on WhatsApp-Shared Radiographs", submitted to *Discover Artificial Intelligence* journal.

## Overview

This study investigates the robustness of AI-assisted diagnostic performance in medical imaging under realistic image compression conditions, specifically on panoramic radiographs shared via WhatsApp. We utilize a MONAI-based deep learning pipeline to evaluate how lossy image compression affects diagnostic reliability, comparing the performance of human evaluators and AI models.

## Repository Structure

*   `data/`: Contains information about the datasets used (not raw data due to privacy and size concerns).
*   `src/`: Contains the source code for the MONAI pipeline, model training, and evaluation scripts.
    *   `src/monai_pipeline.py`: Main script for the MONAI deep learning pipeline.
    *   `src/evaluation_scripts.py`: Scripts for diagnostic performance evaluation (human and AI).
*   `notebooks/`: Jupyter notebooks for data exploration, results visualization, and additional analyses.
*   `models/`: Pre-trained models (if applicable and permitted).
*   `results/`: Evaluation results, including metrics and visualizations.

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/bbvizzotto-run/monai-wpp-images.git
    cd Discover_AI_Robustness_Study
    ```
2.  **Set Up Environment:**
    Install the necessary dependencies (Python 3.x, PyTorch, MONAI, etc.). A `requirements.txt` file will be provided in the `src/` folder.
3.  **Run the MONAI Pipeline:**
    Follow the instructions in `src/monai_pipeline.py` to train or infer with the model.
4.  **Perform Evaluations:**
    Use the scripts in `src/evaluation_scripts.py` to reproduce the performance analyses.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or suggestions, please contact Bruno Boessio Vizzotto (bbvizzotto@ufg.br).
