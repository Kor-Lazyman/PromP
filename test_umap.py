import umap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def to_list(df):
    temp_lst = []
    for x in range(len(df)):
        temp = []
        for y in range(1, len(df.iloc[x])):
            temp.append(df.iloc[x][y])
        temp_lst.append(temp)
    return temp_lst
after = []
before = []
models = ["Random", "VPG_MAML", "ProMP"]
for model in models:
    after.append(pd.read_csv(f"C:/Users/User/Desktop/meta_RL_Fixed/ver4_upgrade/src/ProMP/envs/FEW_SHOT_RESULT/Train_4/{model}_after.csv"))
    before.append(pd.read_csv(f"C:/Users/User/Desktop/meta_RL_Fixed/ver4_upgrade/src/ProMP/envs/FEW_SHOT_RESULT/Train_4/{model}_before.csv"))

# UMAP 모델 생성 (2D로 축소)
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.99, n_components=2, metric="euclidean" , random_state=42)

# Prepare a list to store scatter data for seaborn
scatter_data = []

for x in range(3):
    temp_action_list_before = to_list(before[x])
    temp_action_list_after = to_list(after[x])

    after_map = umap_model.fit_transform(temp_action_list_after)
    before_map = umap_model.fit_transform(temp_action_list_before)

    sns.scatterplot(after_map[:, 0], after_map[:, 1],label = f"{models[x]}_after", alpha=0.5)
    sns.scatterplot(before_map[:, 0], before_map[:, 1],label = f"{models[x]}_before", alpha=0.5)

plt.title("UMAP Projection with Seaborn")
plt.legend()
plt.show()
