import numpy as np
from sklearn.manifold import TSNE
from utils import scale, DATA_PATH

aadata = DATA_PATH + "aadata.csv"

amino_features = []
amino_acids = []

# Amino acid data parsing.
with open(aadata, "r") as data_file:
    lines = [line.replace("\n", "") for line in data_file][1:]
    for line in lines:
        data = line.split(",")

        amino_features.append(
            [float(value) for value in data[1:]]
        )

        amino_acids.append(data[0])

amino_features = np.array(amino_features)

# TSNE dimensionality reduction.
# Number of desired features for each amino acid. Must be <4.
n_components = 3

tsne = TSNE(n_components = n_components, perplexity = len(amino_features) - 1)
tsne_result = tsne.fit_transform(amino_features)

feature_dict = {}
for i in range(len(amino_acids)):
    feature_dict[amino_acids[i]] = tsne_result[i]

print("Dimensionality reduced from {} to {}.".format(amino_features.shape, tsne_result.shape))

# New features.
print(feature_dict)

# Extract each feature to independent dict
feature_dict_1 = {}
feature_dict_2 = {}
feature_dict_3 = {}
 
for key in feature_dict:
    feature_dict_1[key] = feature_dict[key][0]
    feature_dict_2[key] = feature_dict[key][1]
    feature_dict_3[key] = feature_dict[key][2]

print(feature_dict_1)
print(feature_dict_2)
print(feature_dict_3)

# Scale features
scale(feature_dict_1, 1)
scale(feature_dict_2, 1)
scale(feature_dict_3, 1)

print(feature_dict_1)
print(feature_dict_2)
print(feature_dict_3)
  
# Save features
np.save(DATA_PATH + 'TSNE_SP_1.npy', np.array(feature_dict_1)) 
np.save(DATA_PATH + 'TSNE_SP_2.npy', np.array(feature_dict_2)) 
np.save(DATA_PATH + 'TSNE_SP_3.npy', np.array(feature_dict_1)) 