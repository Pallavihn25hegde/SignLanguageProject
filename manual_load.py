# manual_load.py
import os

# UPDATE THIS PATH WITH YOUR ACTUAL PATH
PHRASE_DATASET_PATH = r"E:\SignLanguageProject\_frames"  # <--- CHANGE THIS

print(f"Loading from: {PHRASE_DATASET_PATH}")

if os.path.exists(PHRASE_DATASET_PATH):
    print("✅ Path exists!")
    print("\nContents:")
    for item in os.listdir(PHRASE_DATASET_PATH):
        if '_frames' in item:
            print(f"  📁 {item}")
    
    # Save the path
    with open('phrase_dataset_path.txt', 'w') as f:
        f.write(PHRASE_DATASET_PATH)
    
    print(f"\n✅ Path saved to 'phrase_dataset_path.txt'")
    print("\nNow run: python load_phrase_dataset_corrected.py")
else:
    print(f"❌ Path does not exist: {PHRASE_DATASET_PATH}")
    print("\nPlease update the PHRASE_DATASET_PATH variable with the correct path")