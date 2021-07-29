import glob
import pickle
import random

keys = glob.glob("../pickle/*.pkl")
keys = [key.split("/")[-1] for key in keys]
random.shuffle(keys)
length = len(keys)
with open("train_keys.pkl", "wb") as w:
    pickle.dump(keys[length//5:], w)
with open("val_keys.pkl", "wb") as w:
    pickle.dump(keys[:length//10], w)
with open("test_keys.pkl", "wb") as w:
    pickle.dump(keys[length//10:length//5], w)
