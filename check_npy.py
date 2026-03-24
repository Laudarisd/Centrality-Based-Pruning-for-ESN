import numpy as np

path = "./data/traffic.npy"  # change path
arr = np.load(path)

print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("ndim:", arr.ndim)
print("first rows:")
print(arr[:5])

# If it's 2D and you want the training column currently used:
if arr.ndim >= 2:
    series = arr[:, 1] if arr.shape[1] > 1 else arr[:, 0]
    print("series preview:", series[:10])
