import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.model_selection import train_test_split

def load_and_split_data(data_path, test_size=0.2, random_state=42):
    """
    Load data from a jsonl file and split into train and test sets.
    
    Parameters:
    -----------
    data_path : str
        Path to the jsonl file
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    train_data : list
        Training data
    test_data : list
        Testing data
    """
    # Load all data
    all_data = []
    with open(data_path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    
    # Split into train and test
    train_data, test_data = train_test_split(
        all_data, test_size=test_size, random_state=random_state, 
        stratify=[item['label_int'] for item in all_data]  # Stratify by label to maintain class distribution
    )
    
    print(f"Loaded {len(all_data)} samples")
    print(f"Training set: {len(train_data)} samples")
    print(f"Testing set: {len(test_data)} samples")
    
    return train_data, test_data

def save_split_data(train_data, test_data, train_path, test_path):
    """
    Save the split data to jsonl files.
    """
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(test_path, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved training data to {train_path}")
    print(f"Saved testing data to {test_path}")

class NaiveTextDetector:
    """
    A naive baseline model for AI-generated text detection.
    This model predicts based on the most frequent class in the training data or
    simple statistics derived from the loss values.
    """
    
    def __init__(self, strategy='majority'):
        """
        Initialize the NaiveTextDetector.
        
        Parameters:
        -----------
        strategy : str
            The strategy to use for prediction.
            'majority': Always predict the most frequent class
            'mean_loss': Use mean loss values for prediction
            'min_loss': Predict based on minimum loss (model with lowest perplexity)
        """
        self.strategy = strategy
        self.label_distribution = None
        self.most_common_label = None
        self.label_means = None
    
    def fit(self, train_data_path):
        """
        Fit the naive model on training data.
        
        Parameters:
        -----------
        train_data_path : str
            Path to the training data file (jsonl format)
        """
        # Load and parse the training data
        labels = []
        losses_by_label = {}
        
        with open(train_data_path, 'r') as f:
            for line in tqdm(f, desc="Loading training data"):
                data = json.loads(line)
                label = data['label_int']
                losses = data['losses']
                
                labels.append(label)
                
                if label not in losses_by_label:
                    losses_by_label[label] = []
                losses_by_label[label].append(losses)
        
        # Calculate label distribution and most common label
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.label_distribution = dict(zip(unique_labels, counts))
        self.most_common_label = max(self.label_distribution, key=self.label_distribution.get)
        
        # Calculate mean loss values for each label
        self.label_means = {}
        for label, loss_list in losses_by_label.items():
            self.label_means[label] = np.mean(loss_list, axis=0)
        
        print(f"Label distribution: {self.label_distribution}")
        print(f"Most common label: {self.most_common_label}")
        
        return self
    
    def predict(self, test_data_path):
        """
        Make predictions on test data.
        
        Parameters:
        -----------
        test_data_path : str
            Path to the test data file (jsonl format)
        
        Returns:
        --------
        predictions : list
            Predicted labels
        true_labels : list
            True labels from the test data
        """
        predictions = []
        true_labels = []
        
        with open(test_data_path, 'r') as f:
            for line in tqdm(f, desc="Making predictions"):
                data = json.loads(line)
                true_label = data['label_int']
                losses = data['losses']
                
                if self.strategy == 'majority':
                    # Always predict the most common label from the training set
                    prediction = self.most_common_label
                
                elif self.strategy == 'mean_loss':
                    # Calculate the mean of losses
                    mean_loss = np.mean(losses)
                    
                    # Find the label with the closest mean loss
                    closest_label = None
                    min_distance = float('inf')
                    
                    for label, label_mean in self.label_means.items():
                        mean_label_loss = np.mean(label_mean)
                        distance = abs(mean_loss - mean_label_loss)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_label = label
                    
                    prediction = closest_label
                
                elif self.strategy == 'min_loss':
                    # Predict the label based on which model has the lowest loss
                    # This assumes losses array index corresponds to model index
                    # For example, losses[0] is for model 0 (gpt2), losses[1] is for model 1 (llama), etc.
                    prediction = np.argmin(losses)
                
                predictions.append(prediction)
                true_labels.append(true_label)
        
        return predictions, true_labels
    
    def evaluate(self, y_true, y_pred, label_map=None):
        """
        Evaluate the model performance.
        
        Parameters:
        -----------
        predictions : list
            Predicted labels
        true_labels : list
            True labels
        
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        if label_map is None:
            label_map = {0: 'gpt2', 1: 'llama', 2: 'human', 3: 'gpt3re'}
        
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Get per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        
        print("\nDetailed Metrics by Label:")
        print("---------------------------")
        print(f"{'Label':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 50)
        
        for label in sorted(label_map.keys()):
            label_name = label_map[label]
            if str(label) in report:
                metrics = report[str(label)]
                print(f"{label_name:<10} {metrics['precision']:.4f}      {metrics['recall']:.4f}      {metrics['f1-score']:.4f}      {metrics['support']:<10}")
        
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Print confusion matrix with labels
        cm_display = pd.DataFrame(conf_matrix, 
                                index=[label_map[i] for i in range(len(label_map))], 
                                columns=[label_map[i] for i in range(len(label_map))])
        print(cm_display)
        
        # Also create and save a heatmap visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[label_map[i] for i in range(len(label_map))],
                    yticklabels=[label_map[i] for i in range(len(label_map))])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(f'confusion_matrix.png')
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }


def main():
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Define paths to data
    train_data_path = 'dataset/train_data.jsonl'
    test_data_path = 'dataset/test_data.jsonl'

    # If you haven't already split the data
    train_data, test_data = load_and_split_data('dataset/en_all.jsonl')
    save_split_data(train_data, test_data, train_data_path, test_data_path)
    
    # Initialize and evaluate models with different strategies
    strategies = ['majority', 'mean_loss', 'min_loss']
    results = {}
    
    for strategy in strategies:
        print(f"\n===== Strategy: {strategy} =====")
        model = NaiveTextDetector(strategy=strategy)
        model.fit(train_data_path)
        predictions, true_labels = model.predict(test_data_path)
        metrics = model.evaluate(predictions, true_labels)
        results[strategy] = metrics
    
    # Compare strategies
    print("\n===== Strategy Comparison =====")
    for strategy, metrics in results.items():
        print(f"{strategy}: Accuracy = {metrics['accuracy']:.4f}, Macro F1 = {metrics['macro_f1']:.4f}")

    # And when determining the best strategy, consider both metrics:
    best_accuracy_strategy = max(results, key=lambda x: results[x]['accuracy'])
    best_f1_strategy = max(results, key=lambda x: results[x]['macro_f1'])

    print(f"\nBest strategy by accuracy: {best_accuracy_strategy} with accuracy {results[best_accuracy_strategy]['accuracy']:.4f}")
    print(f"Best strategy by macro F1: {best_f1_strategy} with macro F1 {results[best_f1_strategy]['macro_f1']:.4f}")

if __name__ == "__main__":
    main()