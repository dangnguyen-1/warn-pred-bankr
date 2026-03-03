from preprocess import (
    load_data,
    rename_columns,
    create_binary_label,
    first_failure_filter,
    engineer_features,
)

from model import (
    get_feature_cols,
    split_holdout,
    run_walk_forward_cv,
    evaluate_holdout,
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
)

from visualization import plot_grouped_metrics


def print_metrics(title, metrics):
    print(title)
    print("precision:", round(metrics["precision"], 6))
    print("recall:   ", round(metrics["recall"], 6))
    print("f1:       ", round(metrics["f1"], 6))
    print("confusion_matrix:\n", metrics["confusion_matrix"])
    if "n_folds_used" in metrics:
        print("folds_used:", metrics["n_folds_used"])
    print()


def main():
    df = load_data("data/american_bankruptcy.csv")
    df = rename_columns(df)
    df = create_binary_label(df)
    df = first_failure_filter(df)
    df = engineer_features(df)

    feature_cols = get_feature_cols(df)

    train_df, test_df = split_holdout(df, train_end_year=2014, test_start_year=2015)

    models = [
        ("logreg", train_logistic_regression),
        ("rf", train_random_forest),
        ("gb", train_gradient_boosting),
        ("xgb", train_xgboost),
    ]

    # Collect metrics for plotting
    cv_metrics_by_model = {}
    holdout_metrics_by_model = {}

    # Stage 1: Walk-forward CV on 1999–2014
    print("=== Walk-Forward CV (1999–2014) ===\n")
    for name, train_fn in models:
        _, avg = run_walk_forward_cv(
            train_df,
            train_fn=train_fn,
            feature_cols=feature_cols,
            start_year=1999,
            end_year=2014,
            min_train_years=7,
            threshold=0.05,
        )
        cv_metrics_by_model[name] = avg
        print_metrics(f"{name} (avg over folds)", avg)

    # Stage 2: Final holdout test on 2015–2018
    print("=== Final Holdout (Train: 1999–2014, Test: 2015–2018) ===\n")
    for name, train_fn in models:
        _, test_metrics = evaluate_holdout(
            train_df=train_df,
            test_df=test_df,
            train_fn=train_fn,
            feature_cols=feature_cols,
            threshold=0.05,
        )
        holdout_metrics_by_model[name] = test_metrics
        print_metrics(f"{name} (holdout)", test_metrics)

    plot_grouped_metrics(cv_metrics_by_model, holdout_metrics_by_model)
    
    
if __name__ == "__main__":
    main()