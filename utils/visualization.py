import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, fbeta_score
from sklearn.tree import plot_tree
import numpy as np
import shap

def plot_decision_tree_rules(model, feature_names, save_path=None):
    """Visualiza un Árbol de Decisión."""
    fig, ax = plt.subplots(figsize=(25, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=['No Valoración', 'Sí Valoración'],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        proportion=True
    )
    plt.title('Árbol de Decisión — Reglas de Clasificación (max_depth=5)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_roc_pr_curves(results_dict, y_test, save_path=None):
    """
    Plotea curvas ROC y PR comparativas.
    results_dict: dict {model_name: {'y_proba': array, ...}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Curva ROC ---
    ax = axes[0]
    for name, res in results_dict.items():
        y_proba = res['y_proba']
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('Curva ROC', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # --- Curva Precision-Recall ---
    ax = axes[1]
    for name, res in results_dict.items():
        y_proba = res['y_proba']
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(recall_vals, precision_vals, label=f'{name}', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Curva Precision-Recall', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(results_dict, y_test, save_path=None):
    """Grid de matrices de confusión."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1: axes = [axes]
    
    for ax, (name, res) in zip(axes, results_dict.items()):
        y_pred = res['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['Real 0', 'Real 1'])
        ax.set_title(f'{name}', fontsize=11)
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicción')
    
    plt.suptitle('Matrices de Confusión — Test Set', fontsize=14, y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_threshold_optimization(thresholds, f2_scores, optimal_threshold, save_path=None):
    """Grafica la evolución del F2-Score según el umbral."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f2_scores, linewidth=2, color='steelblue')
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Óptimo: {optimal_threshold:.2f}')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default: 0.50')
    ax.set_xlabel('Umbral de Decisión', fontsize=12)
    ax.set_ylabel('F2-Score', fontsize=12)
    ax.set_title('Optimización del Umbral de Decisión', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_shap_summary(shap_values, X, save_path=None):
    """Wrapper para SHAP summary plot."""
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X, max_display=25, show=False)
    plt.title('SHAP — Importancia Global de Features (Top 25)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_shap_dependence(shap_values, X, feature_names_list, save_path=None):
    """Plots top 5 dependence plots."""
    fig, axes = plt.subplots(1, len(feature_names_list), figsize=(5*len(feature_names_list), 4))
    if len(feature_names_list) == 1: axes = [axes]
    
    for ax, feat in zip(axes, feature_names_list):
        shap.dependence_plot(
            feat, shap_values, X,
            ax=ax, show=False
        )
        ax.set_title(feat, fontsize=10)
    
    plt.suptitle('SHAP Dependence Plots — Top Features', fontsize=14, y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_shap_waterfall(shap_values, explainer, X, index, save_path=None, expected_value=None):
    """Plots waterfall for a specific instance."""
    shap.initjs()
    
    base_val = expected_value
    if base_val is None:
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
             base_val = base_val[1]
    
    try:
        base_val = float(base_val)
    except:
        pass

    plt.figure(figsize=(14, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[index],
            base_values=base_val,
            data=X.iloc[index],
            feature_names=list(X.columns)
        ),
        max_display=15,
        show=False
    )
    plt.title(f'SHAP Waterfall — Paciente #{index}', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
