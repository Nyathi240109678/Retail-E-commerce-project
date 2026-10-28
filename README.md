
<h1 align="center">
  RetailIQ
  <br>
  AI-Powered Retail Sales Forecasting
</h1>

<p align="center">
  Smarter sales predictions for data-driven retail decisions.
</p>

<div align="center">
  <img src="https://img.shields.io/badge/Version-1.0.0-orange" alt="Version 1.0.0">
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-blue" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Python-Programming-yellow?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Artificial%20Intelligence-AI-white" alt="Artificial Intelligence">
  <img src="https://img.shields.io/badge/Retail-E--Commerce-purple" alt="Retail and E-Commerce">
</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [Project Objectives](#project-objectives)
- [Machine Learning Approach](#machine-learning-approach)
- [Technologies Used](#technologies-used)
- [Model Evaluation](#model-evaluation)
- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
- [How to Run the Project](#how-to-run-the-project)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [PowerPoint Presentation](#powerpoint-presentation)
- [Limitations](#limitations)

---

## Project Overview

RetailIQ is an AI-powered retail sales forecasting project developed in Python. It uses machine learning to estimate the sales amount of a retail transaction based on selected transaction details.

The project focuses on the retail and e-commerce industry, where businesses handle transactions across different sales channels and regions. By using historical retail data, RetailIQ demonstrates how machine learning can support data-driven business decisions.

The project also includes a retail assistant chatbot that provides responses to common retail-related questions.

## Problem Statement

Retail businesses process transactions involving different products, prices, discounts, payment methods, sales channels and regions. Without a data-driven approach, estimating transaction values and understanding sales patterns can be difficult.

RetailIQ addresses this problem by developing a machine learning model that estimates the sales amount of a transaction using relevant input features.

## Proposed Solution

RetailIQ uses a supervised machine learning regression model to predict the expected sales amount for a retail transaction.

The system uses historical retail transaction data to train the model. After training, the model can accept user-provided transaction details and produce an estimated sales amount.

The project also includes a retail assistant chatbot to support users with retail-related questions.

## Project Objectives

- Develop a functional machine learning model for retail sales amount prediction.
- Apply data cleaning and preprocessing techniques to historical transaction data.
- Use regression to estimate transaction sales amounts.
- Evaluate model performance using appropriate regression metrics.
- Provide a simple interface for users to obtain predictions.
- Demonstrate how AI can support decision-making in retail and e-commerce.

## Machine Learning Approach

**Learning type:** Supervised learning

**Problem type:** Regression

**Algorithm:** Multiple Linear Regression

**Target variable:** `sales_amount`

The dataset is prepared by handling missing values, converting transaction dates into useful date-related features and encoding categorical variables.

The dataset is split into training and testing sets using a 60% training and 40% testing split, with `random_state=42` for reproducibility.

The model is saved using Joblib so that it can be loaded later by the prediction application.

### Input Features

The documented model features include:

- Quantity
- Unit price
- Discount percentage
- Transaction year
- Transaction month
- Day of the month
- Day of the week
- Sales channel
- Payment method
- Region

The exact features must match those used by the final training script and saved model.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib | Visualizing actual and predicted values |
| Joblib | Saving and loading the trained model |
| NLTK | Natural Language Processing for the chatbot, if used in the final implementation |
| Git and GitHub | Version control and project hosting |

## Model Evaluation

The model is evaluated using regression metrics:

- **R-squared (R²):** Measures how well the model explains variation in the target variable.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
- **Root Mean Squared Error (RMSE):** Measures prediction error while giving greater weight to larger errors.

An actual-versus-predicted scatter plot is also used to help assess model performance.

The final report should include the actual evaluation scores produced by the completed training script.

## Key Features

- Retail transaction sales amount prediction.
- Data preprocessing and categorical feature encoding.
- Reusable trained machine learning model.
- Saved model features for consistent prediction inputs.
- Retail assistant chatbot.
- Model evaluation and visualization.
- Python-based implementation.

## Installation and Setup

### Prerequisites

Install Python and Git on your computer.

### 1. Clone the repository

```bash
git clone https://github.com/Nyathi240109678/Retail-E-commerce-project.git
```

### 2. Open the project folder

```bash
cd Retail-E-commerce-project
```

### 3. Install the required libraries

```bash
pip install pandas numpy scikit-learn matplotlib joblib nltk
```

Install only the libraries required by the final scripts. If NLTK is used, complete any required NLTK data setup specified by the chatbot code.

## How to Run the Project

### Run the Retail Sales Forecasting Model

```bash
python sales_predictor.py
```

The prediction program loads the saved model and feature list and allows the user to enter transaction information to obtain an estimated sales amount.

### Run the Retail Assistant Chatbot

```bash
python Retail_assistant_chatbot.py
```

The chatbot provides responses to supported retail-related questions.

### Run the Model Training and Evaluation Script

```bash
python retail_regression.py
```

Run this script when training or evaluating the model is required. Confirm that its data path and output filenames match the project setup.

## Project Structure

| File | Description |
|---|---|
| `README.md` | Project introduction and setup guide |
| `retail_regression.py` | Model training and evaluation script |
| `sales_predictor.py` | Loads the trained model and performs predictions |
| `Retail_assistant_chatbot.py` | Retail assistant chatbot |
| `Retail_sales_dataset.csv` | Historical retail transaction dataset |
| `sales_model.pkl` | Saved trained machine learning model |
| `model_features.pkl` | Saved feature names used by the model |
| `1. Business Background` | Business background document |
| `5. Business Objectives` | Business objectives document |
| `Retail-Sales-Forecasting-ML-Model-DOCUMENTATION` | Main project documentation |
| `poster 3.pdf` | Project poster |
| `PowerPoint_Presentation` | Project PowerPoint presentation |
| `LICENSE` | Project licensing information |

## Documentation

For more information, consult the project documentation and poster included in this repository.

- Business background
- Business objectives
- Retail Sales Forecasting ML Model documentation
- Project poster

## PowerPoint Presentation

The PowerPoint presentation provides an overview of the RetailIQ project, including its problem statement, objectives, machine learning approach and proposed solution.

[View the PowerPoint Presentation](PowerPoint_Presentation)

---

## Limitations

- Prediction quality depends on the quality and representativeness of the training data.
- Predictions are estimates and may differ from actual transaction values.
- The model estimates transaction sales amounts; it does not directly forecast future inventory demand.
- Changes in customer behaviour, prices or retail conditions may affect prediction accuracy.
- The saved model and prediction application must use compatible input features.

---
