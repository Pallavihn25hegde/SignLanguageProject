import os
import subprocess
import sys

def print_menu():
    print("\n" + "="*50)
    print("SIGN LANGUAGE RECOGNITION - PHRASE INTEGRATION")
    print("="*50)
    print("1. Collect Phrase Data")
    print("2. Prepare Combined Dataset")
    print("3. Train Phrase Model")
    print("4. Evaluate Phrase Model")
    print("5. Run Real-time Recognition")
    print("6. Complete Workflow (All Steps)")
    print("7. Exit")
    print("="*50)

def run_command(command):
    """Run a command and handle errors"""
    try:
        subprocess.run([sys.executable, command], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {command}: {e}")
        return False

def main():
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            print("\n=== Collecting Phrase Data ===")
            run_command('collect_phrase_landmarks.py')
            
        elif choice == '2':
            print("\n=== Preparing Combined Dataset ===")
            run_command('prepare_combined_dataset.py')
            
        elif choice == '3':
            print("\n=== Training Phrase Model ===")
            run_command('train_phrase_model.py')
            
        elif choice == '4':
            print("\n=== Evaluating Phrase Model ===")
            run_command('evaluate_phrase_model.py')
            
        elif choice == '5':
            print("\n=== Running Real-time Recognition ===")
            run_command('realtime_phrase_predict.py')
            
        elif choice == '6':
            print("\n=== Running Complete Workflow ===")
            steps = [
                ('collect_phrase_landmarks.py', "Collecting phrase data"),
                ('prepare_combined_dataset.py', "Preparing dataset"),
                ('train_phrase_model.py', "Training model"),
                ('realtime_phrase_predict.py', "Starting recognition")
            ]
            
            for step_file, step_desc in steps:
                print(f"\n--- {step_desc} ---")
                input("Press Enter to continue...")
                if not run_command(step_file):
                    print(f"Failed at: {step_desc}")
                    break
                    
        elif choice == '7':
            print("\nGoodbye!")
            break
            
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    # Check if required directories exist
    os.makedirs('dataset/phrases', exist_ok=True)
    print("✅ Directories created successfully")
    main()