import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_loss_curve(losses, title, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, labels, title, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(models_data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    model_names = [m['name'] for m in models_data]
    accuracies = [m['accuracy'] for m in models_data]
    precisions = [m['precision'] for m in models_data]
    recalls = [m['recall'] for m in models_data]
    f1_scores = [m['f1'] for m in models_data]
    
    axes[0, 0].bar(model_names, accuracies, color='skyblue')
    axes[0, 0].set_title('Accuracy', fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    
    axes[0, 1].bar(model_names, precisions, color='lightgreen')
    axes[0, 1].set_title('Precision', fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].bar(model_names, recalls, color='lightcoral')
    axes[1, 0].set_title('Recall', fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    
    axes[1, 1].bar(model_names, f1_scores, color='lightyellow')
    axes[1, 1].set_title('F1-Score', fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
