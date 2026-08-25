# check_letter_structure.py
import pickle
import numpy as np

print("=" * 60)
print("CHECKING LETTER DATA STRUCTURE")
print("=" * 60)

with open('dataset/letters/data.pickle', 'rb') as f:
    letter_data = pickle.load(f)

print(f"\nType of data: {type(letter_data)}")
print(f"Keys: {letter_data.keys() if isinstance(letter_data, dict) else 'Not a dict'}")

if isinstance(letter_data, dict):
    for key, value in letter_data.items():
        print(f"\nKey: {key}")
        print(f"  Type: {type(value)}")
        if isinstance(value, list):
            print(f"  Length: {len(value)}")
            print(f"  First element type: {type(value[0])}")
            if len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    print(f"  Shape of first element: {value[0].shape}")
        elif isinstance(value, np.ndarray):
            print(f"  Shape: {value.shape}")
            