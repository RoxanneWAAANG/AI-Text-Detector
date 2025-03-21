# Non Deep Learning Method: Random Forest

![maxresdefault](https://github.com/user-attachments/assets/6456882c-bd0e-49ea-b99a-d9e208e4e8b1)
*image from Kaggle

# Steps

## Step 1
Data is converted into a BoW via a Count Vectorizer 

## Step 2
The vectors are then transformed via a TF-IDF transformation

## Step 3
The transformations are then used to train and fit a Random Forest Classification Model

# Overall Metrics

| Metric      | Value   |
|-------------|---------|
| Accuracy    | 48.8%   |
| Precision   | 47.6%   |
| Recall      | 48.86%  |
| F1 Score    | 46.45%  |

# Per Class Metrics 

| Metric     | GPT 2   | GPT 3   | Human   | Llama   |
|------------|---------|---------|---------|---------|
| Accuracy   | 46.29%  | 89.55%  | 24.03%  | 35.59%  |
| Precision  | 39.02%  | 63.19%  | 49.75%  | 38.46%  |
| Recall     | 46.29%  | 89.55%  | 24.03%  | 35.59%  |
| F1 Score   | 42.35%  | 74.09%  | 32.4%   | 36.97%  |

# Confusion Matrix
|           | GPT 2 | GPT 3 | Human | Llama |
|-----------|--------|--------|--------|--------|
| **GPT 2** |  556   |  221   |  123   |  301   |
| **GPT 3** |   66   | 1071   |   31   |   28   |
| **Human** |  385   |  224   |  296   |  327   |
| **Llama** |  418   |  179   |  145   |  410   |
