import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Define the Labels
labels = [
    "air_cond.", "car_horn", "child_play", "dog_bark", 
    "drilling", "eng_idle", "gun_shot", "jackhammer", 
    "siren", "street_music"
]

# 2. Define the Data (extracted from your LaTeX tables)

# Matrix for AST-P
cm_astp = np.array(
     [[463 ,   3 ,  78 , 127 ,  12 , 181 ,   1 ,   0 ,  33 , 102],
   [4 , 324 ,   5 ,  26 ,   5 ,   0 ,   0 ,   2 ,  32 ,  31],
   [2 ,   0 , 955 ,   9 ,   7 ,   1 ,   0 ,   0 ,  11 ,  15],
   [8 ,   0 ,  20 , 957 ,   0 ,   0 ,   2 ,   0 ,   6 ,   7],
  [55 ,   0 ,  12 ,  28 , 773 ,  68 ,   1 ,   6 ,  15 ,  42],
  [28 ,   4 ,  32 ,  12 ,  11 , 860 ,   0 ,   6 ,  28 ,  19],
   [4 ,   0 ,   0 ,  10 ,   0 ,   0 , 360 ,   0 ,   0 ,   0],
  [44 ,   0 ,   4 ,  17 , 104 , 212 ,   3 , 544 ,  17 ,  55],
   [1 ,   2 ,  13 ,  10 ,   0 ,   0 ,   0 ,   0 , 891 ,  12],
   [1 ,   4 , 106 ,   9 ,   0 ,   0 ,   0 ,   1 ,  22 , 857]]
)

# Matrix for AST-S
cm_asts = np.array(
    [ [677 ,   1 ,  14 ,  55 ,  18 ,  56 ,   0 ,   1 ,  27 , 151],
   [8 , 303 ,   1 ,  10 ,  15 ,   1 ,   0 ,   1 ,  37 ,  53],
   [6 ,   1 , 861 ,  46 ,   8 ,   2 ,   2 ,   0 ,  21 ,  53],
  [21 ,   5 ,  38 , 888 ,   9 ,   6 ,   4 ,   0 ,  13 ,  16],
 [119 ,  14 ,  10 ,  12 , 714 ,  17 ,  19 ,  11 ,  21 ,  63],
 [314 ,   4 ,  36 ,  26 ,  11 , 519 ,   1 ,  21 ,  21 ,  47],
  [10 ,   1 ,   0 ,   7 ,   2 ,   0 , 354 ,   0 ,   0 ,   0],
 [172 ,   5 ,   3 ,   1 ,  59 , 140 ,  14 , 493 ,   0 , 113],
  [12 ,   7 ,   9 ,  16 ,  17 ,   0 ,   1 ,  25 , 800 ,  42],
  [14 ,   9 ,  49 ,   6 ,   3 ,   0 ,   0 ,   0 ,  29 , 890]]
)

# 3. Plotting
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Common heatmap arguments
heatmap_args = {
    'annot': True, 
    'fmt': 'd', 
    'cmap': 'Blues', 
    'xticklabels': labels, 
    'yticklabels': labels,
    'cbar': False
}

# Plot AST-P
sns.heatmap(cm_astp, ax=axes[0], **heatmap_args)
axes[0].set_title('Confusion Matrix: AST-P', fontsize=16)
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Plot AST-S
sns.heatmap(cm_asts, ax=axes[1], **heatmap_args)
axes[1].set_title('Confusion Matrix: AST-S', fontsize=16)
axes[1].set_xlabel('Predicted Label', fontsize=12)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()