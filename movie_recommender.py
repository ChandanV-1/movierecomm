"""
Movie Recommendation using Collaborative Filtering (ALS) in PySpark
Dataset: MovieLens ml-latest-small (https://grouplens.org/datasets/movielens/)

Setup:
    pip install pyspark matplotlib pandas
    Unzip the dataset so that ./ml-latest-small/ratings.csv and movies.csv exist.
Run:
    python movie_recommender_als.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

DATA_DIR = "ml-latest-small"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- 1. Spark
spark = (SparkSession.builder
         .appName("MovieRecommenderALS")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------- 2. Load
ratings = (spark.read.csv(f"{DATA_DIR}/ratings.csv", header=True, inferSchema=True)
           .select("userId", "movieId", F.col("rating").cast("float")))
movies = spark.read.csv(f"{DATA_DIR}/movies.csv", header=True, inferSchema=True)

ratings.cache()
n_ratings = ratings.count()
n_users = ratings.select("userId").distinct().count()
n_movies = ratings.select("movieId").distinct().count()
sparsity = 1 - n_ratings / (n_users * n_movies)
print(f"Ratings: {n_ratings} | Users: {n_users} | Movies: {n_movies} | Sparsity: {sparsity:.4f}")

# ---------------------------------------------------------------- 3. EDA (Spark SQL)
ratings.createOrReplaceTempView("ratings")
movies.createOrReplaceTempView("movies")

print("\nRating distribution:")
dist = spark.sql("SELECT rating, COUNT(*) AS cnt FROM ratings GROUP BY rating ORDER BY rating")
dist.show()

print("Top 10 most-rated movies:")
spark.sql("""
    SELECT m.title, COUNT(*) AS num_ratings, ROUND(AVG(r.rating), 2) AS avg_rating
    FROM ratings r JOIN movies m ON r.movieId = m.movieId
    GROUP BY m.title ORDER BY num_ratings DESC LIMIT 10
""").show(truncate=False)

print("Top 10 highest-rated movies (min 50 ratings):")
spark.sql("""
    SELECT m.title, COUNT(*) AS num_ratings, ROUND(AVG(r.rating), 2) AS avg_rating
    FROM ratings r JOIN movies m ON r.movieId = m.movieId
    GROUP BY m.title HAVING COUNT(*) >= 50
    ORDER BY avg_rating DESC, num_ratings DESC LIMIT 10
""").show(truncate=False)

print("Ratings per user (summary):")
ratings.groupBy("userId").count().select("count").describe().show()

# Plot: rating distribution
pdf = dist.toPandas()
plt.figure(figsize=(7, 4))
plt.bar(pdf["rating"].astype(str), pdf["cnt"], color="#3b82f6")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rating_distribution.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 4. Split
train, test = ratings.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train.count()} | Test: {test.count()}")

# ---------------------------------------------------------------- 5. ALS + CV
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop",   # avoid NaN RMSE for unseen users/items
          nonnegative=True, seed=42)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

param_grid = (ParamGridBuilder()
              .addGrid(als.rank, [10, 30, 50])
              .addGrid(als.regParam, [0.05, 0.1, 0.3])
              .addGrid(als.maxIter, [10])
              .build())

cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid,
                    evaluator=evaluator, numFolds=3, parallelism=2, seed=42)

print("\nTraining ALS with 3-fold cross-validation ...")
cv_model = cv.fit(train)
best = cv_model.bestModel
print(f"Best params -> rank={best.rank}, "
      f"regParam={best._java_obj.parent().getRegParam()}, "
      f"maxIter={best._java_obj.parent().getMaxIter()}")

# CV results plot
cv_rmse = cv_model.avgMetrics
labels = [f"r={p[als.rank]},λ={p[als.regParam]}" for p in param_grid]
plt.figure(figsize=(9, 4))
plt.bar(range(len(cv_rmse)), cv_rmse, color="#10b981")
plt.xticks(range(len(cv_rmse)), labels, rotation=45, ha="right")
plt.ylabel("CV RMSE")
plt.title("Hyperparameter Tuning Results")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cv_results.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 6. Evaluate
pred = best.transform(test)
rmse = evaluator.evaluate(pred)
mae = RegressionEvaluator(metricName="mae", labelCol="rating",
                          predictionCol="prediction").evaluate(pred)
print(f"\nTest RMSE: {rmse:.4f} | Test MAE: {mae:.4f}")

# Baseline: always predict global mean rating
mean_rating = train.agg(F.avg("rating")).first()[0]
baseline = test.withColumn("prediction", F.lit(float(mean_rating)))
print(f"Baseline (global mean) RMSE: {evaluator.evaluate(baseline):.4f}")

# ---------------------------------------------------------------- 7. Recommend
print("\nTop-5 recommendations for 5 sample users:")
user_recs = best.recommendForAllUsers(5)
flat = (user_recs
        .select("userId", F.explode("recommendations").alias("rec"))
        .select("userId", F.col("rec.movieId").alias("movieId"),
                F.round("rec.rating", 2).alias("predicted_rating"))
        .join(movies, "movieId")
        .select("userId", "title", "genres", "predicted_rating"))
flat.filter(F.col("userId") <= 5).orderBy("userId", F.desc("predicted_rating")).show(25, truncate=False)

# Save recommendations for all users
flat.toPandas().to_csv(f"{OUT_DIR}/recommendations.csv", index=False)

# Users who would like a specific movie
print("Top-5 users for each of 3 sample movies:")
best.recommendForAllItems(5).filter(F.col("movieId").isin(1, 50, 296)).show(truncate=False)

# ---------------------------------------------------------------- 8. Save model
best.write().overwrite().save(f"{OUT_DIR}/als_model")
print(f"\nDone. Outputs saved in ./{OUT_DIR}/")
spark.stop()