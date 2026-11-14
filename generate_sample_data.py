"""Generate a synthetic Instagram-style dataset so the project runs without
downloading anything. Replace data/social_media.csv with your real dataset
(same column names) to analyse real data."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 120

home = rng.integers(1500, 12000, n)
hashtags = rng.integers(500, 9000, n)
explore = rng.integers(200, 15000, n)
other = rng.integers(50, 2500, n)
impressions = home + hashtags + explore + other

df = pd.DataFrame({
    "Impressions": impressions,
    "From Home": home,
    "From Hashtags": hashtags,
    "From Explore": explore,
    "From Other": other,
    "Saves": (impressions * rng.uniform(0.01, 0.06, n)).astype(int),
    "Comments": rng.integers(0, 30, n),
    "Shares": (impressions * rng.uniform(0.0, 0.01, n)).astype(int),
    "Likes": (impressions * rng.uniform(0.02, 0.06, n)).astype(int),
    "Profile Visits": (impressions * rng.uniform(0.003, 0.02, n)).astype(int),
    "Follows": rng.integers(0, 40, n),
    "Caption": ["Learn data analysis with Python"] * n,
    "Hashtags": [
        " ".join(rng.choice(
            ["#python", "#data", "#analytics", "#machinelearning", "#coding",
             "#datascience", "#ai", "#statistics", "#programming", "#tech"],
            size=5, replace=False)) for _ in range(n)
    ],
})
os.makedirs("data", exist_ok=True)
df.to_csv("data/social_media.csv", index=False)
print("Saved data/social_media.csv", df.shape)
