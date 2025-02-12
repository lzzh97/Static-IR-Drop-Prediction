# Saliency-Based Explainability for Static IR Drop Prediction 

_Supported by NSF award 2322713_

This repository contains the implementation of the methodologies proposed in the paper "[Static IR Drop Prediction with Attention U-Net and Saliency-Based Explainability](https://www.arxiv.org/abs/2408.03292)" by Lizi Zhang and Azadeh Davoodi. The project focuses on improving static IR drop prediction in power delivery networks (PDNs) using a novel Attention U-Net model and providing explainability of predictions through saliency maps.

## Introduction

Static IR drop analysis is a critical task in integrated circuit design, as it helps ensure that power delivery networks (PDNs) provide stable and sufficient voltage across a chip. Traditional methods for IR drop analysis can be computationally expensive. This project introduces AttUNet, a neural network model that leverages attention mechanisms to predict static IR drop efficiently and accurately. Additionally, the project includes a method for generating saliency maps to explain the predicted IR drop and identify the root causes of high-drop regions.

## Features

  - **AttUNet Model**: A U-Net-based architecture with embedded attention gates for enhanced prediction accuracy.
  - **Pretrain-Finetune Strategy**: Utilizes artificially generated data for pretraining and fine-tunes on limited real design data to prevent overfitting.
  - **Data Augmentation**: Includes a variety of image transformations to increase the robustness of the model.
  - **Saliency Maps**: Provides a fast and interpretable method to identify the key contributors to predicted IR drop, facilitating targeted optimizations.

## Installation
To install the required dependencies, clone this repository and install the necessary Python packages:

```bash
git clone https://github.com/lzzh97/Static-IR-Drop-Prediction.git
cd Static-IR-Drop-Prediction
pip install -r requirements.txt
```

Ensure you have Python 3.8 or later installed.

## Data Preperation
The dataset required for this project is included in the repository. This data was originally provided by the ICCAD 2023 contest (https://github.com/ASU-VDA-Lab/ML-for-IR-drop) and is used for training and evaluating the models in this repository.

## Usage

### Training and Evaluation

1. Run the pretraining phase using the artificially generated dataset.
```
python ./AttUNet/train_attunet.py --phase pretrain
```

2. Fine-tune the model using a smaller, real dataset.
```
python ./AttUNet/train_attunet.py --phase finetune --pre <path to pretrained model>
```

3. Evaluate the model on the test dataset.
```
python ./AttUNet/evaluate.py --model <path to model>
```

4. Generate saliency maps to explain and diagnose high IR drop predictions.
```
python ./AttUNet/generate_saliency_maps.py --model <path to model>
```


