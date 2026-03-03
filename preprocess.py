import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def rename_columns(df):
    column_mapping = {"X1": "current_assets", "X2": "cost_of_goods_sold", "X3": "depreciation_amortization", 
                      "X4": "ebitda", "X5": "inventory", "X6": "net_income", "X7": "total_receivables", 
                      "X8": "market_cap", "X9": "net_sales", "X10": "total_assets", "X11": "long_term_debt", 
                      "X12": "ebit", "X13": "gross_profit", "X14": "current_liabilities", "X15": "retained_earnings",
                      "X16": "total_revenue", "X17": "total_liabilities", "X18": "operating_expenses"}

    df = df.rename(columns = column_mapping)
    return df


def create_binary_label(df):
    df["status_label"] = (df["status_label"] == "failed").astype(int)
    return df


def first_failure_filter(df):
    df = df.sort_values(["company_name", "year"])

    def keep_until_failure(group):
        if 1 in group["status_label"].values:
            first_fail_year = group.loc[group["status_label"] == 1, "year"].min()
            group = group[group["year"] <= first_fail_year]
        return group

    df = df.groupby("company_name", group_keys = False).apply(keep_until_failure)
    return df


def engineer_features(df):
    df["net_profit_margin"] = df["net_income"] / df["total_revenue"].replace(0, 1e-6)
    df["gross_profit_margin"] = df["gross_profit"] / df["net_sales"].replace(0, 1e-6)

    df["roa"] = df["net_income"] / df["total_assets"].replace(0, 1e-6)
    df["ros"] = df["net_income"] / df["net_sales"].replace(0, 1e-6)

    df["current_ratio"] = df["current_assets"] / df["current_liabilities"].replace(0, 1e-6)
    df["quick_ratio"] = (df["current_assets"] - df["inventory"]) / df["current_liabilities"].replace(0, 1e-6)

    df["debt_to_asset_ratio"] = df["total_liabilities"] / df["total_assets"].replace(0, 1e-6)
    df["asset_turnover"] = df["net_sales"] / df["total_assets"].replace(0, 1e-6)

    ratio_cols = ["net_profit_margin", "gross_profit_margin", 
                  "roa", "ros", "current_ratio", "quick_ratio", 
                  "debt_to_asset_ratio", "asset_turnover"]

    for col in ratio_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    return df