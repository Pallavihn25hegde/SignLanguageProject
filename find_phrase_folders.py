# find_phrase_folders.py
import os

def find_phrase_folders():
    print("Searching for phrase folders...")
    print("=" * 50)
    
    # Common locations to search
    search_paths = [
        os.path.expanduser("~"),  # Home directory
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Downloads"),
        "C:\\",
        "D:\\",
        os.getcwd(),  # Current directory
    ]
    
    found_folders = []
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        print(f"\nSearching in: {search_path}")
        try:
            for root, dirs, files in os.walk(search_path):
                for dir_name in dirs:
                    if '_frames' in dir_name and dir_name not in ['asl_env', 'dataset']:
                        full_path = os.path.join(root, dir_name)
                        found_folders.append(full_path)
                        print(f"  ✓ Found: {full_path}")
                        
                # Limit search depth to avoid scanning entire drive
                if root.count(os.sep) - search_path.count(os.sep) > 3:
                    break
        except PermissionError:
            continue
    
    if found_folders:
        print("\n" + "=" * 50)
        print(f"Found {len(found_folders)} phrase folders:")
        for folder in found_folders:
            print(f"  - {folder}")
        
        # Save the path for later use
        with open('phrase_folders_path.txt', 'w') as f:
            for folder in found_folders:
                f.write(folder + '\n')
        
        print("\n✅ Paths saved to 'phrase_folders_path.txt'")
    else:
        print("\n❌ No '_frames' folders found!")
        print("\nPlease manually check these locations:")
        print("1. Your Downloads folder")
        print("2. Your Desktop")
        print("3. The folder where you extracted the dataset")

if __name__ == "__main__":
    find_phrase_folders()