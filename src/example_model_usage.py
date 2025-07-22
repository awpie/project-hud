# Example script demonstrating modular model usage
from execution.inference_current import run_inference_with_model
from model_clipZeroShot import CLIPZeroShotModel
from model_mock import MockModel

def main():
    """Demonstrate how to use different models with the same inference loop."""
    
    print("=== Activity Classification Model Selection ===")
    print("1. CLIP Zero-shot Model (real inference)")
    print("2. Mock Model (demonstration)")
    
    while True:
        try:
            choice = input("\nSelect model (1 or 2): ").strip()
            
            if choice == '1':
                print("\nInitializing CLIP Zero-shot Model...")
                model = CLIPZeroShotModel(buffer_size=30)
                print("✓ CLIP model loaded successfully!")
                break
                
            elif choice == '2':
                print("\nInitializing Mock Model...")
                model = MockModel()
                print("✓ Mock model loaded successfully!")
                break
                
            else:
                print("Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
            return
        except Exception as e:
            print(f"Error: {e}")
            return
    
    # Run inference with the selected model
    print(f"\nStarting inference with {type(model).__name__}...")
    run_inference_with_model(model)

if __name__ == "__main__":
    main() 