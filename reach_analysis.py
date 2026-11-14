"""Social Media Reach Analysis
Explores what drives post reach (impressions) and builds a simple
model to predict it.

Usage:
    python reach_analysis.py                    # uses data/social_media.csv
    python reach_analysis.py --data my_file.csv
"""
import argparse
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

OUT = "outputs"
SOURCES = ["From Home", "From Hashtags", "From Explore", "From Other"]
ENGAGEMENT = ["Saves", "Comments", "Shares", "Likes", "Profile Visits", "Follows"]


def load_data(path):
    df = pd.read_csv(path, encoding="latin1")
    df = df.drop_duplicates().dropna(subset=["Impressions"])
    print(f"Loaded {df.shape[0]} posts, {df.shape[1]} columns")
    return df


def save(fig_name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fig_name), dpi=150)
    plt.close()


def summary(df):
    print("\n=== Summary statistics ===")
    print(df[["Impressions"] + SOURCES + ENGAGEMENT].describe().round(1))
    print("\n=== Share of total impressions by source ===")
    share = (df[SOURCES].sum() / df[SOURCES].sum().sum() * 100).round(1)
    print(share.astype(str) + " %")


def plots(df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Impressions"], kde=True, color="#4c72b0")
    plt.title("Distribution of Impressions")
    save("impressions_distribution.png")

    plt.figure(figsize=(6, 6))
    df[SOURCES].sum().plot.pie(autopct="%1.1f%%", startangle=90)
    plt.ylabel("")
    plt.title("Where Impressions Come From")
    save("impression_sources.png")

    plt.figure(figsize=(9, 7))
    corr = df[["Impressions"] + SOURCES + ENGAGEMENT].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save("correlation_heatmap.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Likes", "Saves", "Profile Visits"]):
        sns.regplot(data=df, x=col, y="Impressions", ax=ax,
                    scatter_kws={"alpha": 0.5})
        ax.set_title(f"Impressions vs {col}")
    save("impressions_vs_engagement.png")


def top_hashtags(df, n=10):
    if "Hashtags" not in df.columns:
        return
    tags = Counter(t.lower() for row in df["Hashtags"].dropna()
                   for t in str(row).split() if t.startswith("#"))
    top = pd.Series(dict(tags.most_common(n)))
    print(f"\n=== Top {n} hashtags ===")
    print(top)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top.values, y=top.index, color="#55a868")
    plt.title("Most Used Hashtags")
    save("top_hashtags.png")


def model(df):
    # Sources are excluded: Impressions is just their sum (data leakage)
    features = [c for c in ENGAGEMENT if c in df.columns]
    X, y = df[features], df["Impressions"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("\n=== Predicting Impressions ===")
    results = {}
    for name, m in {"Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(
                        n_estimators=200, random_state=42)}.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        results[name] = m
        print(f"{name}: R2 = {r2_score(y_test, pred):.3f}, "
              f"MAE = {mean_absolute_error(y_test, pred):.0f}")

    imp = pd.Series(results["Random Forest"].feature_importances_,
                    index=features).sort_values()
    plt.figure(figsize=(8, 5))
    imp.plot.barh(color="#c44e52")
    plt.title("Feature Importance (Random Forest)")
    save("feature_importance.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/social_media.csv")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    df = load_data(args.data)
    summary(df)
    plots(df)
    top_hashtags(df)
    model(df)
    print(f"\nDone. Charts saved in '{OUT}/'.")


if __name__ == "__main__":
    main()
