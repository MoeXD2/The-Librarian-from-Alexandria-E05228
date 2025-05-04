### Setup Instructions

Before you start working on the project, please follow these setup instructions:

1. **Create a virtual environment**  
   In your terminal or command prompt, navigate to the project folder and run:
   ```bash
   python -m venv .venv
   ```
   This will create a virtual environment in the `.venv` folder.

2. **Install dependencies**  
   Once the virtual environment is created, activate it:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   After activation, install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start working on the project**  
   The main code for this project resides in the `notebooks` and `training` folders, where you will find Jupyter notebooks (numbered sequentially starting from `001`) and a python file (for training). These notebooks and file guide you through the project step-by-step. 

4. **Data Folder**  
   The `data` folder contains processed images that have been resized, normalized, and converted to RGB. These processed images are ready to be used in the project to speed up the workflow. It also includes augmented images that are used for training the model. The original images are not included in the repository due to their size. If you need the original images.

5. **Processed Data Information**  
   There is a `pages_processed.csv` file in the repository. This file contains new paths that point to the processed images in the `data` folder, instead of the raw ones. Using these paths will allow for faster processing during the analysis and modeling steps.

## Folder Structure

- `.venv/` - Virtual environment folder.
- `notebooks/` - Jupyter notebooks for the project.
- `training/` - The python file for training the model.
- `results/` - Training and test results.
- `saved_models` - The saved models after training for later use. 
- `data/` - Processed images (resized, normalized, and converted to RGB) and CSV file (`pages_processed.csv`) as well as others with updated image paths pointing to the processed images, and augmented images used for training the model.
- `requirements.txt` - List of required libraries for the project.

## Notes

- You need Python 3.10 if you want to use DirectML, otherwise the code will not run (or it will run but not use the GPU), meaning that the training will be slow.