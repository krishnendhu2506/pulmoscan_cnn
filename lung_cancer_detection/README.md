# Lung Cancer Detection Web Application using CNN

A Flask + TensorFlow application for CT scan classification into Normal, Adenocarcinoma, and Squamous Cell Carcinoma, with PDF medical report generation.

## Project Structure

- `api/app.py`
- `model/train_model.py`
- `model/predict.py`
- `utils/preprocess.py`
- `utils/report_generator.py`
- `templates/`
- `static/`
- `database/patients.db`
- `requirements.txt`
- `vercel.json`

## Dataset Preparation

Your current dataset folders contain extra classes and long class names. Use the provided normalizer to create a clean dataset with only:

- `lung_n`
- `lung_aca`
- `lung_scc`

Command (from `lung_cancer_detection/`):

```bash
python rename_dataset.py --src ..\dataset --dst ..\dataset_std
```

This will copy images into `dataset_std/train`, `dataset_std/valid`, `dataset_std/test` and ignore the large cell carcinoma folders.

## Training the Model

From `lung_cancer_detection/model`:

```bash
python train_model.py --train-dir ..\..\dataset_std\train --valid-dir ..\..\dataset_std\valid --output .\lung_cancer_model.h5 --epochs 12 --batch-size 16
```

The training script also writes `class_indices.json` next to the model so predictions use the correct label order.

## Run Locally

From `lung_cancer_detection/`:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python api\app.py
```

Open `http://127.0.0.1:5000`.

## Notes on Confidence

The app displays the model's softmax probabilities. These values are always between 0 and 100 and sum to 100. Confidence depends on the model and input quality.

## Vercel Deployment

Install Vercel CLI:

```bash
npm install -g vercel
```

Deploy:

```bash
vercel login
vercel
vercel --prod
```

## GitHub Push Commands

```bash
git init
git add .
git commit -m "Initial commit - Lung Cancer Detection AI Web App"
git branch -M main
git remote add origin https://github.com/yourusername/lung-cancer-detection.git
git push -u origin main
```

## Disclaimer

This tool is intended for research and educational purposes only and should not replace professional medical diagnosis.
