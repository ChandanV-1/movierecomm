# Movie Recommendation System using ALS (Apache Spark)
## Overview
A collaborative filtering recommender built with PySpark MLlib's ALS algorithm
on the MovieLens dataset (100,836 ratings, 610 users, 9,724 movies).

## Results
| Model | RMSE | MAE |
|---|---|---|
| Baseline (global mean) | 1.0504 | - |
| ALS (rank=50, regParam=0.1) | 0.8706 | 0.6748 |

ALS improves on the baseline by about 17%.

## Tech Stack
Python, Apache Spark (PySpark), Spark MLlib, Spark SQL, Matplotlib, Pandas

## How to Run
1. Install Java 8/11/17 and Python 3.
2. `pip install -r requirements.txt`
3. Download **ml-latest-small** from https://grouplens.org/datasets/movielens/
   and unzip it in the project root so `ml-latest-small/ratings.csv` exists.
4. `python movie_recommender_als.py`

Charts, recommendations and the trained model are saved in `output/`.

## Project Pipeline
Load data -> Spark SQL EDA -> 80/20 split -> ALS + 3-fold CV -> Evaluate (RMSE, MAE) -> Top-N recommendations
