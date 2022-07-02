import numpy as np
from tqdm import tqdm

# First, randomly generate top-10 document embeddings with dimension of 3.
embeds = np.random.randn(10, 3)
print("Top-10 doc embeddings:\n", embeds, end='\n\n')

# Let's assume the 1st, 2nd, 5th and 7th documents are relevant.
rel_labels = [1, 1, 0, 0, 1, 0, 1, 0, 0, 0]

# Hence, the optimal aggregation embedding should be the following:
optimal_embeds = np.sum(embeds[np.nonzero(rel_labels)[0]], axis=0)
print("Optimal aggregation embedding:", optimal_embeds, end='\n\n')

# Then, we simulate users examination propensity of each rank position:
propensities = np.power(np.divide(1, np.arange(1.0, 11)), 1)
print("User examination propensity of each rank position:\n", propensities, end='\n\n')

# We assume user will click every relevant documents they examined, thus the click probability of each doc is:
click_probs = propensities * rel_labels
print("User click probability of each document:\n", click_probs, end='\n\n')

# Now let's simulate the ranking has been logged 10k times and use two aggregation methods to get click feedback embeddings.
bias_emb = np.zeros(3)
unbias_emb = np.zeros(3)
for i in tqdm(range(10000), desc="Simulating user clicks"):
    rand = np.random.rand(10)
    clicks = rand < click_probs
    clicks = clicks.astype(int)
    bias_emb += np.sum(embeds[np.nonzero(clicks)[0]], axis=0)  # Naive aggregation
    unbias_emb += np.sum((embeds / propensities[:, np.newaxis])[np.nonzero(clicks)[0]], axis=0)  # Unbiased aggregation

# You should observe unbiased aggregation embedding is more close to the optimal aggregation embedding:
print('Biased aggregation embedding:', bias_emb/10000)
print('Unbiased aggregation embedding:', unbias_emb/10000)
print("Optimal aggregation embedding:", optimal_embeds, end='\n\n')
