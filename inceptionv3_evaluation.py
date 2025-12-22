"""
InceptionV3 Model Evaluation and Visualization
ASL Fingerspelling Recognition Project
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os

# Configuration
MODEL_PATH = 'models/inceptionv3_asl.keras'  # Change to .h5 if needed
TEST_DATA_PATH = 'data/asl_alphabet_test'
IMG_SIZE = (299, 299)  # InceptionV3 uses 299x299
BATCH_SIZE = 32

# ASL Alphabet Classes
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'space', 'del', 'nothing']

def load_model():
    """Load the trained InceptionV3 model"""
    print("Loading InceptionV3 model...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    model.summary()
    return model

def prepare_test_data():
    """Prepare test data generator"""
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

def evaluate_model(model, test_generator):
    """Evaluate model and generate metrics"""
    print("\n" + "="*60)
    print("EVALUATING INCEPTIONV3 MODEL")
    print("="*60)
    
    # Predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✓ Overall Accuracy: {accuracy*100:.2f}%")
    
    # Precision, Recall, F1-Score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    print(f"✓ Precision: {precision*100:.2f}%")
    print(f"✓ Recall: {recall*100:.2f}%")
    print(f"✓ F1-Score: {f1*100:.2f}%")
    
    # Classification Report
    print("\n" + "-"*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("-"*60)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print(report)
    
    # Save report
    os.makedirs('results', exist_ok=True)
    with open('results/inceptionv3_classification_report.txt', 'w') as f:
        f.write(f"InceptionV3 Model Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Precision: {precision*100:.2f}%\n")
        f.write(f"Recall: {recall*100:.2f}%\n")
        f.write(f"F1-Score: {f1*100:.2f}%\n\n")
        f.write("-"*60 + "\n")
        f.write(report)
    
    return y_true, y_pred, predictions

def plot_confusion_matrix(y_true, y_pred):
    """Generate and save confusion matrix"""
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Count'})
    plt.title('InceptionV3 - Confusion Matrix', fontsize=20, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/inceptionv3_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved: results/inceptionv3_confusion_matrix.png")
    plt.close()

def plot_accuracy_per_class(y_true, y_pred):
    """Plot per-class accuracy"""
    print("\nGenerating per-class accuracy plot...")
    
    class_accuracy = []
    for i in range(len(CLASSES)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).sum() / mask.sum()
            class_accuracy.append(acc * 100)
        else:
            class_accuracy.append(0)
    
    plt.figure(figsize=(16, 8))
    bars = plt.bar(CLASSES, class_accuracy, color='mediumpurple', edgecolor='indigo')
    
    # Color bars based on accuracy
    for i, bar in enumerate(bars):
        if class_accuracy[i] >= 90:
            bar.set_color('lightgreen')
        elif class_accuracy[i] >= 70:
            bar.set_color('gold')
        else:
            bar.set_color('salmon')
    
    plt.axhline(y=90, color='green', linestyle='--', label='90% threshold', alpha=0.5)
    plt.axhline(y=70, color='orange', linestyle='--', label='70% threshold', alpha=0.5)
    
    plt.title('InceptionV3 - Per-Class Accuracy', fontsize=18, fontweight='bold')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 105])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/inceptionv3_class_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ Per-class accuracy plot saved: results/inceptionv3_class_accuracy.png")
    plt.close()

def plot_top_errors(y_true, y_pred):
    """Identify and visualize most common misclassifications"""
    print("\nAnalyzing top errors...")
    
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]
    
    error_pairs = {}
    for idx in error_indices:
        pair = (y_true[idx], y_pred[idx])
        error_pairs[pair] = error_pairs.get(pair, 0) + 1
    
    top_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    labels = [f"{CLASSES[true]} → {CLASSES[pred]}" for (true, pred), _ in top_errors]
    counts = [count for _, count in top_errors]
    
    bars = ax.barh(labels, counts, color='plum', edgecolor='purple')
    ax.set_xlabel('Number of Misclassifications', fontsize=12)
    ax.set_title('InceptionV3 - Top 10 Misclassification Pairs', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f' {int(width)}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/inceptionv3_top_errors.png', dpi=300, bbox_inches='tight')
    print("✓ Top errors plot saved: results/inceptionv3_top_errors.png")
    plt.close()

def plot_confidence_distribution(predictions, y_true, y_pred):
    """Plot confidence score distribution"""
    print("\nGenerating confidence distribution plot...")
    
    max_confidences = np.max(predictions, axis=1)
    correct = y_true == y_pred
    
    plt.figure(figsize=(12, 6))
    plt.hist(max_confidences[correct], bins=50, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(max_confidences[~correct], bins=50, alpha=0.7, label='Incorrect Predictions', color='red')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('InceptionV3 - Confidence Score Distribution', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/inceptionv3_confidence_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Confidence distribution saved: results/inceptionv3_confidence_distribution.png")
    plt.close()

def main():
    """Main execution"""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load model
    model = load_model()
    
    # Prepare data
    test_generator = prepare_test_data()
    
    # Evaluate
    y_true, y_pred, predictions = evaluate_model(model, test_generator)
    
    # Generate visualizations
    plot_confusion_matrix(y_true, y_pred)
    plot_accuracy_per_class(y_true, y_pred)
    plot_top_errors(y_true, y_pred)
    plot_confidence_distribution(predictions, y_true, y_pred)
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE!")
    print("="*60)
    print("All results saved in 'results/' directory")

if __name__ == "__main__":
    main()