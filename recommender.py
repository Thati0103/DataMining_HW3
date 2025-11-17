# Task 2: Movie Recommender Analysis
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data
ratings = pd.read_csv("ratings_small.csv")  # Ensure file is in working directory
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

# Open text file for writing results
with open("task2_results.txt", "w") as f:

    # 2. Probabilistic Matrix Factorization (SVD)
    f.write("=== Probabilistic Matrix Factorization (PMF / SVD) ===\n")
    svd = SVD()
    cv_svd = cross_validate(svd, data, measures=['RMSE','MAE'], cv=5, verbose=True)
    pmf_rmse = np.mean(cv_svd['test_rmse'])
    pmf_mae = np.mean(cv_svd['test_mae'])
    f.write(f"PMF Average RMSE: {pmf_rmse:.6f}\n")
    f.write(f"PMF Average MAE: {pmf_mae:.6f}\n\n")

    # 3. User-based & Item-based Collaborative Filtering
    sim_options = ['cosine', 'msd', 'pearson']
    k_values = [5, 10, 20, 30, 40, 50]

    user_cf_results = {'sim':[], 'k':[], 'rmse':[], 'mae':[]}
    item_cf_results = {'sim':[], 'k':[], 'rmse':[], 'mae':[]}

    for sim in sim_options:
        for k in k_values:
            # User-based CF
            ub_cf = KNNBasic(k=k, sim_options={'name': sim, 'user_based': True})
            cv_ub = cross_validate(ub_cf, data, measures=['RMSE','MAE'], cv=5, verbose=False)
            user_cf_results['sim'].append(sim)
            user_cf_results['k'].append(k)
            user_cf_results['rmse'].append(np.mean(cv_ub['test_rmse']))
            user_cf_results['mae'].append(np.mean(cv_ub['test_mae']))

            # Item-based CF
            ib_cf = KNNBasic(k=k, sim_options={'name': sim, 'user_based': False})
            cv_ib = cross_validate(ib_cf, data, measures=['RMSE','MAE'], cv=5, verbose=False)
            item_cf_results['sim'].append(sim)
            item_cf_results['k'].append(k)
            item_cf_results['rmse'].append(np.mean(cv_ib['test_rmse']))
            item_cf_results['mae'].append(np.mean(cv_ib['test_mae']))

    # Convert results to DataFrames
    user_df = pd.DataFrame(user_cf_results)
    item_df = pd.DataFrame(item_cf_results)

    # 4. Save User/Item CF results
    f.write("=== User-based CF Results ===\n")
    f.write(user_df.to_string(index=False))
    f.write("\n\n=== Item-based CF Results ===\n")
    f.write(item_df.to_string(index=False))
    f.write("\n\n")

    # 5. Identify best K for User/Item-based CF (lowest RMSE)
    best_user = user_df.loc[user_df['rmse'].idxmin()]
    best_item = item_df.loc[item_df['rmse'].idxmin()]
    f.write("=== Best User-based CF ===\n")
    f.write(best_user.to_string())
    f.write("\n\n=== Best Item-based CF ===\n")
    f.write(best_item.to_string())
    f.write("\n")

# 6. Plot RMSE vs K for each similarity metric
plt.figure(figsize=(12,5))
for sim in sim_options:
    subset = user_df[user_df['sim']==sim]
    plt.plot(subset['k'], subset['rmse'], marker='o', label=f'User-CF {sim}')
    subset = item_df[item_df['sim']==sim]
    plt.plot(subset['k'], subset['rmse'], marker='x', linestyle='--', label=f'Item-CF {sim}')
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("RMSE")
plt.title("RMSE vs K for User & Item-based CF (All Similarities)")
plt.legend()
plt.grid()
plt.show()
