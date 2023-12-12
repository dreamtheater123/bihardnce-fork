import pickle
import json

# Load queries
with open('queries', 'rb') as file:
    queries = pickle.load(file)
    # list of all queries

# Load candidates
with open('candidates', 'rb') as file:
    candidates = pickle.load(file)
    # list of all candidates

# Load testing_new
with open('testing_new', 'rb') as file:
    testing_new = pickle.load(file)

# Load json file CHIP-CDN_train.json
with open('../chipcdn_data/CHIP-CDN_train.json', 'r', encoding='utf-8') as file:
    train_data = json.load(file)

pass
