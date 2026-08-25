# search_phrase_folders.py
import os

def search_for_frames():
    print("Searching for '_frames' folders...")
    print("=" * 60)
    
    # Start from E: drive root
    search_drives = ['E:\\', 'C:\\Users']
    found_folders = []
    
    for drive in search_drives:
        if not os.path.exists(drive):
            continue
        
        print(f"\nSearching in: {drive}")
        try:
            for root, dirs, files in os.walk(drive):
                # Look for folders ending with _frames
                for dir_name in dirs:
                    if dir_name.endswith('_frames'):
                        full_path = os.path.join(root, dir_name)
                        found_folders.append(full_path)
                        print(f"  ✓ FOUND: {full_path}")
                
                # Limit search depth to avoid long scan
                depth = root.count(os.sep) - drive.count(os.sep)
                if depth > 4:
                    break
                    
        except PermissionError:
            continue
        except Exception as e:
            print(f"  Error accessing {drive}: {e}")
    
    if found_folders:
        print("\n" + "=" * 60)
        print(f"Found {len(found_folders)} phrase folders:")
        for folder in found_folders:
            print(f"  - {folder}")
        
        # Save paths
        with open('found_phrase_paths.txt', 'w') as f:
            for folder in found_folders:
                f.write(folder + '\n')
        
        print("\n✅ Paths saved to 'found_phrase_paths.txt'")
        
        # Also save the parent directory
        if found_folders:
            parent_dir = os.path.dirname(found_folders[0])
            print(f"\n📁 Parent directory: {parent_dir}")
            print(f"\nRun this command next:")
            print(f'python load_phrase_dataset_corrected.py')
            print(f'Then enter: {parent_dir}')
    else:
        print("\n❌ No '_frames' folders found!")
        print("\nPlease check if you extracted the dataset.")
        print("Common locations to check manually:")
        print("  - E:\\Downloads")
        print("  - E:\\Datasets")
        print("  - Desktop folder")
        print("  - Documents folder")

if __name__ == "__main__":
    search_for_frames()