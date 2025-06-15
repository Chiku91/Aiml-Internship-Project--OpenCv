import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open("output/features/broadcast.pkl", "rb") as f:
    B = pickle.load(f)
with open("output/features/tacticam.pkl", "rb") as f:
    T = pickle.load(f)

matches = []
for t_fn, t_v in T.items():
    sims = cosine_similarity([t_v], list(B.values()))[0]
    i = np.argmax(sims)
    b_fn = list(B.keys())[i]
    matches.append((t_fn, b_fn, sims[i]))

matches.sort(key=lambda x: -x[2])
for t_fn, b_fn, sim in matches[:20]:
    print(f"{t_fn} ↔ {b_fn} (cos={sim:.3f})")