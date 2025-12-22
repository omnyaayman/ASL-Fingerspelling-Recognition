"""
Model Comparison Script
Compare ResNet50, EfficientNetB0, and InceptionV3 for ASL Recognition
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
import time
import os

# Configuration
MODELS = {
    'ResNet50': {
        'path': 'models/resnet50_asl.keras',
        'img_size': (224, 224),
        'preprocess': resnet_preprocess
    },
    'EfficientNetB0': {
        'path': 'models/efficientnetb0_asl.keras',
        'img_size': (224, 224),
        'preprocess': efficient_preprocess
    },
    'InceptionV3': {
        'path': 'models/inceptionv3_asl.keras',
        'img_size': (299, 299),
        'preprocess': inception_preprocess
    }
}

TEST_DATA_PATH = 'data/asl_alphabet_test'
BATCH_SIZE = 32

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'space', 'del', 'nothing']

def load_and_evaluate_model(model_name, model_config):
    """Load model and evaluate on test data"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print('='*60)
    
    # Load model
    print(f"Loading {model_name}...")
    model = keras.models.load_model(model_config['path'])
    print("✓ Model loaded")
    
    # Prepare test data
    print("Preparing test data...")
    test_datagen = ImageDataGenerator(preprocessing_function=model_config['preprocess'])
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_PATH,
        target_size=model_config['img_size'],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate
    print("Running predictions...")
    start_time = time.time()
    predictions = model.predict(test_generator, verbose=1)
    inference_time = time.time() - start_time
    
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Calculate model size
    model_size = os.path.getsize(model_config['path']) / (1024 * 1024)  # MB
    
    # Inference speed
    num_samples = len(y_true)
    fps = num_samples / inference_time
    
    results = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'inference_time': inference_time,
        'fps': fps,
        'model_size': model_size,
        'predictions': predictions,
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    print(f"\n✓ {model_name} Results:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Precision: {results['precision']:.2f}%")
    print(f"  Recall:    {results['recall']:.2f}%")
    print(f"  F1-Score:  {results['f1_score']:.2f}%")
    print(f"  FPS:       {results['fps']:.2f}")
    print(f"  Size:      {results['model_size']:.2f} MB")
    
    return results

def plot_comparison(all_results):
    """Create comprehensive comparison visualizations"""
    os.makedirs('results/comparison', exist_ok=True)
    
    models = list(all_results.keys())
    
    # 1. Metrics Comparison
    print("\nGenerating metrics comparison...")
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_labels))
    width = 0.25
    
    for i, model in enumerate(models):
        values = [all_results[model][m] for m in metrics]
        ax.bar(x + i*width, values, width, label=model)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('results/comparison/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: metrics_comparison.png")
    plt.close()
    
    # 2. Performance vs Size Trade-off
    print("Generating performance vs size plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sizes = [all_results[m]['model_size'] for m in models]
    accuracies = [all_results[m]['accuracy'] for m in models]
    fps_values = [all_results[m]['fps'] for m in models]
    
    scatter = ax.scatter(sizes, accuracies, s=[f*10 for f in fps_values], 
                        alpha=0.6, c=range(len(models)), cmap='viridis')
    
    for i, model in enumerate(models):
        ax.annotate(f"{model}\n({fps_values[i]:.1f} FPS)", 
                   (sizes[i], accuracies[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Model Size (MB)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Performance vs Model Size Trade-off', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison/performance_vs_size.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_vs_size.png")
    plt.close()
    
    # 3. Inference Speed Comparison
    print("Generating inference speed comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fps_values = [all_results[m]['fps'] for m in models]
    colors = ['#4a90e2', '#50c878', '#9b59b6']
    
    bars = ax.barh(models, fps_values, color=colors)
    ax.set_xlabel('Frames Per Second (FPS)', fontsize=12)
    ax.set_title('Inference Speed Comparison', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f' {width:.1f} FPS', va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('results/comparison/inference_speed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: inference_speed.png")
    plt.close()
    
    # 4. Detailed Comparison Table
    print("Generating comparison table...")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Metric', 'ResNet50', 'EfficientNetB0', 'InceptionV3'])
    
    metrics_display = [
        ('Accuracy (%)', 'accuracy'),
        ('Precision (%)', 'precision'),
        ('Recall (%)', 'recall'),
        ('F1-Score (%)', 'f1_score'),
        ('FPS', 'fps'),
        ('Model Size (MB)', 'model_size'),
        ('Inference Time (s)', 'inference_time')
    ]
    
    for label, key in metrics_display:
        row = [label]
        for model in models:
            value = all_results[model][key]
            if 'Time' in label:
                row.append(f"{value:.2f}")
            elif 'Size' in label:
                row.append(f"{value:.2f}")
            elif 'FPS' in label:
                row.append(f"{value:.1f}")
            else:
                row.append(f"{value:.2f}")
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4a90e2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    for i in range(1, len(table_data)):
        values = [float(table_data[i][j]) for j in range(1, 4)]
        best_idx = values.index(max(values)) + 1
        
        if i != 6 and i != 7:  # Not size or time (lower is better)
            table[(i, best_idx)].set_facecolor('#90EE90')
        else:
            best_idx = values.index(min(values)) + 1
            table[(i, best_idx)].set_facecolor('#90EE90')
    
    plt.title('Comprehensive Model Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('results/comparison/comparison_table.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_table.png")
    plt.close()

def save_comparison_report(all_results):
    """Save detailed comparison report"""
    print("\nSaving comparison report...")
    
    with open('results/comparison/comparison_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("ASL FINGERSPELLING RECOGNITION - MODEL COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"\n{model_name}\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy:       {results['accuracy']:.2f}%\n")
            f.write(f"Precision:      {results['precision']:.2f}%\n")
            f.write(f"Recall:         {results['recall']:.2f}%\n")
            f.write(f"F1-Score:       {results['f1_score']:.2f}%\n")
            f.write(f"FPS:            {results['fps']:.2f}\n")
            f.write(f"Model Size:     {results['model_size']:.2f} MB\n")
            f.write(f"Inference Time: {results['inference_time']:.2f} seconds\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Best model for each metric
        best_accuracy = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        best_speed = max(all_results.items(), key=lambda x: x[1]['fps'])
        best_size = min(all_results.items(), key=lambda x: x[1]['model_size'])
        
        f.write(f"Best Accuracy:  {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.2f}%)\n")
        f.write(f"Best Speed:     {best_speed[0]} ({best_speed[1]['fps']:.2f} FPS)\n")
        f.write(f"Smallest Size:  {best_size[0]} ({best_size[1]['model_size']:.2f} MB)\n")
    
    print("✓ Saved: comparison_report.txt")

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("ASL FINGERSPELLING RECOGNITION - MODEL COMPARISON")
    print("="*70)
    
    os.makedirs('results/comparison', exist_ok=True)
    
    # Evaluate all models
    all_results = {}
    for model_name, config in MODELS.items():
        try:
            results = load_and_evaluate_model(model_name, config)
            all_results[model_name] = results
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
    
    if len(all_results) == 0:
        print("\n✗ No models could be evaluated!")
        return
    
    # Generate comparisons
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)
    
    plot_comparison(all_results)
    save_comparison_report(all_results)
    
    print("\n" + "="*70)
    print("✓ COMPARISON COMPLETE!")
    print("="*70)
    print("\nResults saved in: results/comparison/")
    print("  - metrics_comparison.png")
    print("  - performance_vs_size.png")
    print("  - inference_speed.png")
    print("  - comparison_table.png")
    print("  - comparison_report.txt")

if __name__ == "__main__":
    main()