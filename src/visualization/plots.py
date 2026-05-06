"""Visualization functions for all required figures."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.evaluation.calibration import expected_calibration_error

SAVE_DIR = "reports/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("Set2", 8)
MODEL_NAMES = {"text_only": "Text-Only", "metadata_only": "Metadata-Only", "multimodal": "Multimodal"}
MODEL_COLORS = {"text_only": COLORS[0], "metadata_only": COLORS[1], "multimodal": COLORS[2]}


def save_fig(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# 1. Class distribution
def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts().sort_index()
    labels_map = {0: "Negative", 1: "Positive"}
    ax.bar([labels_map[i] for i in counts.index], counts.values, color=[COLORS[3], COLORS[0]])
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, str(v), ha="center", fontweight="bold")
    save_fig(fig, "class_distribution.png")


# 2. Agreement distribution
def plot_agreement_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["agreement", "weak_disagreement", "medium_disagreement", "strong_disagreement"]
    counts = df["disagreement_group"].value_counts().reindex(order).fillna(0)
    ax.bar(range(len(counts)), counts.values, color=COLORS[:4])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(["Agreement", "Weak\nDisagreement", "Medium\nDisagreement", "Strong\nDisagreement"])
    ax.set_ylabel("Count")
    ax.set_title("Agreement / Disagreement Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(int(v)), ha="center", fontweight="bold")
    save_fig(fig, "agreement_distribution.png")


# 3. Model accuracy by group
def plot_accuracy_by_group(group_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_groups = ["overall", "agreement", "all_disagreement", "strong_disagreement"]
    data = group_df[group_df["group"].isin(plot_groups)]

    x = np.arange(len(plot_groups))
    width = 0.25
    for i, (model, label) in enumerate(MODEL_NAMES.items()):
        model_data = data[data["model"] == model].set_index("group").reindex(plot_groups)
        ax.bar(x + i * width, model_data["accuracy"], width, label=label, color=MODEL_COLORS[model])

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Overall", "Agreement", "All\nDisagreement", "Strong\nDisagreement"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy by Agreement Group")
    ax.legend()
    ax.set_ylim(0, 1.05)
    save_fig(fig, "model_accuracy_by_group.png")


# 4. F1 by group
def plot_f1_by_group(group_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_groups = ["overall", "agreement", "all_disagreement", "strong_disagreement"]
    data = group_df[group_df["group"].isin(plot_groups)]

    x = np.arange(len(plot_groups))
    width = 0.25
    for i, (model, label) in enumerate(MODEL_NAMES.items()):
        model_data = data[data["model"] == model].set_index("group").reindex(plot_groups)
        ax.bar(x + i * width, model_data["f1"], width, label=label, color=MODEL_COLORS[model])

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Overall", "Agreement", "All\nDisagreement", "Strong\nDisagreement"])
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by Agreement Group")
    ax.legend()
    ax.set_ylim(0, 1.05)
    save_fig(fig, "f1_by_group.png")


# 5. Calibration curves
def plot_calibration_curves(df, n_bins=10):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for model, prob_col in [
        ("text_only", "text_prob_positive"),
        ("metadata_only", "meta_prob_positive"),
        ("multimodal", "mm_prob_positive"),
    ]:
        y_true = df["label"].values
        y_prob = df[prob_col].values
        ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(y_true, y_prob, n_bins)
        mask = bin_counts > 0
        ax.plot(bin_confs[mask], bin_accs[mask], "o-",
                label=f"{MODEL_NAMES[model]} (ECE={ece:.3f})", color=MODEL_COLORS[model])

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves (Reliability Diagram)")
    ax.legend(loc="lower right")
    save_fig(fig, "calibration_curves.png")


# 6. Confidence: correct vs incorrect
def plot_confidence_correct_vs_incorrect(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (model, pred_col, conf_col) in zip(axes, [
        ("text_only", "text_pred", "text_confidence"),
        ("metadata_only", "meta_pred", "meta_confidence"),
        ("multimodal", "mm_pred", "mm_confidence"),
    ]):
        correct = df["label"] == df[pred_col]
        ax.hist(df.loc[correct, conf_col], bins=20, alpha=0.6, label="Correct", color=COLORS[0])
        ax.hist(df.loc[~correct, conf_col], bins=20, alpha=0.6, label="Incorrect", color=COLORS[3])
        ax.set_xlabel("Confidence")
        ax.set_title(MODEL_NAMES[model])
        ax.legend()

    axes[0].set_ylabel("Count")
    fig.suptitle("Confidence Distribution: Correct vs Incorrect Predictions", y=1.02)
    fig.tight_layout()
    save_fig(fig, "confidence_correct_vs_incorrect.png")


# 7. Modality dominance
def plot_modality_dominance(dominance_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = dominance_df["category"].values
    pcts = dominance_df["percentage"].values
    colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[4]]
    bars = ax.bar(cats, pcts, color=colors[:len(cats)])
    ax.set_ylabel("Percentage of Disagreement Cases")
    ax.set_title("Modality Dominance in Disagreement Cases")
    ax.set_xticklabels(["Text\nDominant", "Metadata\nDominant", "Both\nAgree", "Neither"], rotation=0)
    for bar, v in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontweight="bold")
    save_fig(fig, "modality_dominance.png")


# 8. High-confidence errors by group
def plot_high_confidence_errors(df, threshold=0.90):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    groups = ["agreement", "weak_disagreement", "medium_disagreement", "strong_disagreement"]

    for ax, (model, pred_col, conf_col) in zip(axes, [
        ("text_only", "text_pred", "text_confidence"),
        ("metadata_only", "meta_pred", "meta_confidence"),
        ("multimodal", "mm_pred", "mm_confidence"),
    ]):
        rates = []
        for g in groups:
            subset = df[df["disagreement_group"] == g]
            if len(subset) == 0:
                rates.append(0)
                continue
            high_conf = subset[conf_col] >= threshold
            incorrect = subset["label"] != subset[pred_col]
            hce = (high_conf & incorrect).sum()
            hc_total = high_conf.sum()
            rates.append(hce / hc_total * 100 if hc_total > 0 else 0)

        ax.bar(range(len(groups)), rates, color=MODEL_COLORS[model])
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["Agree", "Weak", "Medium", "Strong"], rotation=0)
        ax.set_title(MODEL_NAMES[model])

    axes[0].set_ylabel("High-Confidence Error Rate (%)")
    fig.suptitle("High-Confidence Error Rate by Disagreement Group", y=1.02)
    fig.tight_layout()
    save_fig(fig, "high_confidence_errors.png")


# 9. Confusion matrices
def plot_confusion_matrices(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (model, pred_col) in zip(axes, [
        ("text_only", "text_pred"),
        ("metadata_only", "meta_pred"),
        ("multimodal", "mm_pred"),
    ]):
        cm = confusion_matrix(df["label"], df[pred_col])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"])
        ax.set_title(MODEL_NAMES[model])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Confusion Matrices", y=1.02)
    fig.tight_layout()
    save_fig(fig, "confusion_matrices.png")


# 10. Disagreement severity trend
def plot_disagreement_severity_trend(group_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    severity_order = ["agreement", "weak_disagreement", "medium_disagreement", "strong_disagreement"]
    severity_labels = ["Agreement", "Weak", "Medium", "Strong"]

    for model, label in MODEL_NAMES.items():
        model_data = group_df[
            (group_df["model"] == model) & (group_df["group"].isin(severity_order))
        ].set_index("group").reindex(severity_order)

        error_rate = 1 - model_data["accuracy"].values
        ax.plot(severity_labels, error_rate, "o-", label=label, color=MODEL_COLORS[model], linewidth=2)

    ax.set_xlabel("Disagreement Severity")
    ax.set_ylabel("Error Rate")
    ax.set_title("Error Rate by Disagreement Severity")
    ax.legend()
    save_fig(fig, "disagreement_severity_trend.png")
