import pickle

with open("model.pkl", "rb") as f:
    obj = pickle.load(f)

print(type(obj))
print(obj)
