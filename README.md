# Static IR Drop Prediction with Attention U-Net and Saliency-Based Explainability

This repository contains the implementation of the methodologies proposed in the paper "Static IR Drop Prediction with Attention U-Net and Saliency-Based Explainability" by Lizi Zhang and Azadeh Davoodi. The project focuses on improving static IR drop prediction in power delivery networks (PDNs) using a novel Attention U-Net model and enhancing the explainability of predictions through saliency maps.

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
git clone 
cd Static-IR-Drop-Prediction
pip install -r requirements.txt
