"""

Functions used to evaluate and visualise performance

- Alex Bott


"""
from sklearn.metrics import accuracy_score, f1_score


def evaluate_performance(predictions, true_labels):
        
    # Flatten the predictions and true labels
    flattened_prediction = predictions.ravel()
    flattened_true_labels = true_labels.ravel()
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(flattened_true_labels, flattened_prediction)
    f1 = f1_score(flattened_true_labels, flattened_prediction)
    
    # Print the scores
    print('Accuracy:', accuracy)
    print('F1 Score:', f1)
