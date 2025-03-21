# AI-Text-Detector
## Naive Approach - Mean Loss

### Method
Step 1: Calculate the mean loss for each label in training data

Step 2: Predict the label whose mean loss is closest to the test example's loss for each test example 

## Result
Overall Accuracy: 41.93% 

- Classification Metrics
  
    | Label  | Precision | Recall | F1-Score | Support |
    |--------|-----------|--------|----------|---------|
    | gpt2   | 0.3292    | 0.3206 | 0.3248   | 1232.0  |
    | llama  | 0.2337    | 0.3833 | 0.2904   | 720.0   |
    | human  | 0.4629    | 0.4520 | 0.4574   | 1228.0  |
    | gpt3re | 0.6489    | 0.4866 | 0.5561   | 1599.0  |


- Confusion Matrix

    |         | gpt2 | llama | human | gpt3re |
    |---------|------|-------|-------|--------|
    | gpt2    | 395  | 292   | 265   | 280    |
    | llama   | 134  | 276   | 214   | 96     |
    | human   | 142  | 486   | 555   | 45     |
    | gpt3re  | 529  | 127   | 165   | 778    |
