"""Evaluation helpers for UNISTRA NLP 2026 workshop notebooks."""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(10, 8), title="Confusion Matrix"):
    """Plot a confusion matrix with nice formatting."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels or sorted(set(y_true)),
                yticklabels=labels or sorted(set(y_true)), ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def get_errors(X_test, y_true, y_pred, y_proba=None, n=5, text_col='text'):
    """Get the n most confident misclassifications."""
    mask = y_true != y_pred
    errors = pd.DataFrame({
        'text': X_test.values[mask] if hasattr(X_test, 'values') else X_test[mask],
        'true_label': np.array(y_true)[mask],
        'predicted_label': np.array(y_pred)[mask],
    })
    if y_proba is not None:
        max_proba = np.max(y_proba, axis=1)[mask]
        errors['confidence'] = max_proba
        errors = errors.sort_values('confidence', ascending=False)
    errors['text_preview'] = errors['text'].str[:120] + '...'
    return errors.head(n)


def compare_models(results_dict):
    """Create a comparison DataFrame from a dict of {name: accuracy}."""
    df = pd.DataFrame([
        {'Model': name, 'Accuracy': acc}
        for name, acc in results_dict.items()
    ]).sort_values('Accuracy', ascending=False).reset_index(drop=True)
    return df


def print_classification_summary(y_true, y_pred, model_name="Model"):
    """Print accuracy and classification report with a header."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"  {model_name} — Accuracy: {acc:.1%}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc
