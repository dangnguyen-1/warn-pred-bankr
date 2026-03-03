import matplotlib.pyplot as plt
import numpy as np


def plot_grouped_metrics(cv_metrics_by_model, holdout_metrics_by_model):
    _plot_one_grouped_bar(
        title="Walk-Forward CV Metrics (1999–2014)",
        metrics_by_model=cv_metrics_by_model
    )

    _plot_one_grouped_bar(
        title="Final Holdout Metrics (Train: 1999–2014, Test: 2015–2018)",
        metrics_by_model=holdout_metrics_by_model
    )


def _plot_one_grouped_bar(title, metrics_by_model):
    models = list(metrics_by_model.keys())
    metrics = ["precision", "recall", "f1"]

    values = {m: [metrics_by_model[m][k] for k in metrics] for m in models}

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10, 5))
    for i, metric_name in enumerate(metrics):
        y = [values[m][i] for m in models]
        plt.bar(x + i * width, y, width, label=metric_name)

    plt.xticks(x + width, models)
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()