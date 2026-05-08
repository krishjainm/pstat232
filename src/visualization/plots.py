"""Visualization functions for all required figures."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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


# ---------------------------------------------------------------------------
# New figures for upgrades 3, 5, 6
# ---------------------------------------------------------------------------

# 11. Probability dominance histogram
def plot_probability_dominance_histogram(df):
    """Histogram of probability-based dominance ratio for disagreement cases."""
    disagree = df[df["agreement_status"] == "disagreement"]
    if "dominance_ratio" not in disagree.columns or len(disagree) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(disagree["dominance_ratio"], bins=30, color=COLORS[5], edgecolor="white", alpha=0.8)
    ax.axvline(0.5, color="red", linestyle="--", label="Equal dominance")
    ax.set_xlabel("Dominance Ratio (0=text, 1=metadata)")
    ax.set_ylabel("Count")
    ax.set_title("Probability-Based Dominance Distribution (Disagreement Cases)")
    ax.legend()
    save_fig(fig, "probability_dominance_histogram.png")


# 12. Metadata feature importance
def plot_metadata_feature_importance(feat_imp_df):
    """Bar chart of metadata feature importances."""
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_col = "mean_abs_shap" if "mean_abs_shap" in feat_imp_df.columns else "importance"
    data = feat_imp_df.sort_values(imp_col, ascending=True)
    ax.barh(data["feature"], data[imp_col], color=COLORS[1])
    ax.set_xlabel("Importance" if imp_col == "importance" else "Mean |SHAP|")
    ax.set_title("Metadata Feature Importance")
    fig.tight_layout()
    save_fig(fig, "metadata_feature_importance.png")


# 13. Group-conditional calibration
def plot_calibration_by_group(df, n_bins=10):
    """Calibration curves split by agreement/disagreement for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_specs = [
        ("text_only", "text_prob_positive"),
        ("metadata_only", "meta_prob_positive"),
        ("multimodal", "mm_prob_positive"),
    ]

    for ax, (model, prob_col) in zip(axes, model_specs):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        for grp, color, ls in [("agreement", COLORS[0], "-"), ("disagreement", COLORS[3], "--")]:
            if grp == "agreement":
                mask = df["agreement_status"] == "agreement"
            else:
                mask = df["agreement_status"] == "disagreement"
            subset = df[mask]
            if len(subset) == 0:
                continue
            ece, ba, bc, bcount = expected_calibration_error(
                subset["label"].values, subset[prob_col].values, n_bins
            )
            m = bcount > 0
            ax.plot(bc[m], ba[m], "o-", color=color, linestyle=ls,
                    label=f"{grp.title()} (ECE={ece:.3f})")
        ax.set_title(MODEL_NAMES[model])
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Frequency")
        ax.legend(fontsize=9)

    fig.suptitle("Group-Conditional Calibration Curves", y=1.02)
    fig.tight_layout()
    save_fig(fig, "calibration_by_group.png")


# 14. Selective prediction curves
def plot_selective_prediction(sel_pred_df):
    """Accuracy vs coverage curves for selective prediction."""
    if len(sel_pred_df) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (model, label) in zip(axes, MODEL_NAMES.items()):
        model_data = sel_pred_df[sel_pred_df["model"] == model]
        for grp, color, ls in [("overall", "gray", "-"), ("agreement", COLORS[0], "--"),
                                ("disagreement", COLORS[3], ":")]:
            grp_data = model_data[model_data["group"] == grp].sort_values("coverage")
            if len(grp_data) == 0:
                continue
            ax.plot(grp_data["coverage"], grp_data["accuracy"], ls,
                    color=color, label=grp.title(), linewidth=2)

        ax.set_xlabel("Coverage")
        ax.set_ylabel("Accuracy")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0.5, 1.02)

    fig.suptitle("Selective Prediction: Accuracy vs Coverage", y=1.02)
    fig.tight_layout()
    save_fig(fig, "selective_prediction.png")


# 15. Cross-category comparison
def plot_cross_category_results(cross_cat_df):
    """Bar chart comparing disagreement rates and accuracy across categories."""
    if cross_cat_df is None or len(cross_cat_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cats = cross_cat_df["category"].values

    # Disagreement rates
    if "disagreement_rate" in cross_cat_df.columns:
        axes[0].bar(cats, cross_cat_df["disagreement_rate"] * 100, color=COLORS[3])
        axes[0].set_ylabel("Disagreement Rate (%)")
        axes[0].set_title("Cross-Category Disagreement Rates")
        axes[0].tick_params(axis="x", rotation=30)

    # Multimodal accuracy
    acc_cols = [c for c in cross_cat_df.columns if c.endswith("_accuracy") and "disagree" not in c]
    if acc_cols:
        x = np.arange(len(cats))
        width = 0.25
        for i, col in enumerate(acc_cols):
            label = col.replace("_accuracy", "").replace("_", " ").title()
            axes[1].bar(x + i * width, cross_cat_df[col], width, label=label, color=COLORS[i])
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(cats, rotation=30)
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Model Accuracy by Category")
        axes[1].legend()

    fig.tight_layout()
    save_fig(fig, "cross_category_comparison.png")
