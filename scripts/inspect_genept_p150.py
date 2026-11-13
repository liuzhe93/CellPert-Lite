
import pickle
import numpy as np
import sys

path = sys.argv[1]

with open(path, "rb") as f:
    obj = pickle.load(f)

print("Object type:", type(obj))
print("Dictionary entries:", len(obj))

matches = []

for key, value in obj.items():
    if isinstance(key, str) and key.strip().upper() == "P150":
        vec = np.asarray(value, dtype=np.float32).squeeze()

        matches.append((key, vec))

for key, vec in matches:
    print(
        f"Original key={key!r}, "
        f"shape={vec.shape}, "
        f"finite={np.isfinite(vec).all()}"
    )

for i in range(len(matches)):
    for j in range(i + 1, len(matches)):
        key1, vec1 = matches[i]
        key2, vec2 = matches[j]

        if vec1.shape != vec2.shape:
            print(f"Different dimensions: {key1!r} vs {key2!r}")
        else:
            print(
                f"{key1!r} vs {key2!r}: "
                f"identical={np.array_equal(vec1, vec2)}, "
                f"max_abs_diff={np.max(np.abs(vec1 - vec2)):.6g}"
            )

