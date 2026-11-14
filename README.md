# Social Media Reach Analysis

An internship project that analyses what drives the reach (impressions) of
social media posts and predicts reach using machine learning.

## Objectives
- Understand where impressions come from (Home, Hashtags, Explore, Other)
- Find which engagement metrics (likes, saves, shares, ...) relate to reach
- Identify the most-used hashtags
- Build models (Linear Regression, Random Forest) to predict impressions

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Project structure
```
├── data/                    # CSV dataset
├── outputs/                 # generated charts
├── generate_sample_data.py  # creates a synthetic dataset
├── reach_analysis.py        # main analysis + model
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python generate_sample_data.py     # skip if you have your own CSV
python reach_analysis.py
```
To use a real dataset (e.g. the Instagram Reach dataset on Kaggle), save it as
`data/social_media.csv` or run `python reach_analysis.py --data path/to/file.csv`.

Expected columns: Impressions, From Home, From Hashtags, From Explore,
From Other, Saves, Comments, Shares, Likes, Profile Visits, Follows, Hashtags.

## Results
Charts are saved to `outputs/`: impression distribution, source breakdown,
correlation heatmap, engagement scatter plots, top hashtags and feature
importance.

## Key insights
*(Fill this in after running on your dataset, e.g.)*
- Most impressions come from ___
- ___ has the strongest correlation with reach
- Random Forest achieved R² of ___

## Author
Your Name – Internship Project
