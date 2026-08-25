# find_letter_data.py
import os
import pickle

print("=" * 60)
print("LOCATING LETTER DATASET")
print("=" * 60)

# Check common locations
locations = [
    'data.pickle',
    'dataset/letters/data.pickle',
    'letter_data.pickle',
    'letters.pickle',
    'sign_language_data.pickle'
]

found = False
for loc in locations:
    if os.path.exists(loc):
        print(f"\n✅ Found letter data at: {loc}")
        with open(loc, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                if 'data' in data and 'labels' in data:
                    print(f"   Samples: {len(data['data'])}")
                    print(f"   Classes: {len(set(data['labels']))}")
                    print(f"   Features: {data['data'][0].shape if len(data['data']) > 0 else 'N/A'}")
            found = True
            break

if not found:
    print("\n❌ No letter dataset found in common locations")
    print("\nChecking entire project directory...")
    
    # Search for any pickle file
    for file in os.listdir('.'):
        if file.endswith('.pickle') or file.endswith('.pkl'):
            print(f"  Found: {file}")
            if file != 'unified_data.pickle':
                try:
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                        print(f"    Type: {type(data)}")
                        if isinstance(data, dict):
                            print(f"    Keys: {list(data.keys())[:5]}")
                except:
                    pass