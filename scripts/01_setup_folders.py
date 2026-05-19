import os
import sys

def create_project_structure():
    
    folders = [
        'raw_data/status/no_mask',
        'raw_data/status/proper_mask',
        'raw_data/status/improper_mask',
        'raw_data/type/surgical',
        'raw_data/type/n95',
        'raw_data/type/cloth',
        'raw_data/color/white',
        'raw_data/color/black',
        'raw_data/color/blue',
        
        # Model 1: Mask Status
        'dataset/model1_mask_status/train/no_mask',
        'dataset/model1_mask_status/train/proper_mask',
        'dataset/model1_mask_status/train/improper_mask',
        'dataset/model1_mask_status/test/no_mask',
        'dataset/model1_mask_status/test/proper_mask',
        'dataset/model1_mask_status/test/improper_mask',
        
        # Model 2: Mask Type
        'dataset/model2_mask_type/train/surgical',
        'dataset/model2_mask_type/train/n95',
        'dataset/model2_mask_type/train/cloth',
        'dataset/model2_mask_type/test/surgical',
        'dataset/model2_mask_type/test/n95',
        'dataset/model2_mask_type/test/cloth',
        
        # Model 3: Mask Color
        'dataset/model3_mask_color/train/white',
        'dataset/model3_mask_color/train/black',
        'dataset/model3_mask_color/train/blue',
        'dataset/model3_mask_color/test/white',
        'dataset/model3_mask_color/test/black',
        'dataset/model3_mask_color/test/blue',
        
        # ============ OUTPUT FOLDERS ============
        'models',
        'results/confusion_matrices',
        'results/predictions',
        'utils',
        'docs',
        'outputs'
    ]
    
    print("=" * 70)
    print("SETTING UP PROJECT FOLDER STRUCTURE")
    print("=" * 70)
    
    created_count = 0
    
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            created_count += 1
            print(f"✓ Created: {folder}")
        except Exception as e:
            print(f"✗ Error creating {folder}: {e}")
            return False
    
    print("\n" + "=" * 70)
    print(f"✅ SUCCESS: {created_count} folders created/verified")
    print("=" * 70)
    
    # Print summary
    print("\n📁 FOLDER STRUCTURE SUMMARY:")
    print("\n[RAW DATA - You populate these]")
    print("  raw_data/status/       → no_mask, proper_mask, improper_mask")
    print("  raw_data/type/         → surgical, n95, cloth")
    print("  raw_data/color/        → white, black, blue")
    
    print("\n[PROCESSED DATA - Scripts populate these]")
    print("  dataset/model1_mask_status/  → train/test split for Model 1")
    print("  dataset/model2_mask_type/    → train/test split for Model 2")
    print("  dataset/model3_mask_color/   → train/test split for Model 3")
    
    print("\n[OUTPUT]")
    print("  models/                → Saved trained models (.pkl files)")
    print("  results/               → Metrics, confusion matrices, predictions")
    print("  outputs/               → Inference results")
    
    return True

if __name__ == "__main__":
    success = create_project_structure()
    
    if not success:
        print("\n❌ Failed to create folder structure")
        sys.exit(1)
    else:
        print("\n💡 Next step: Run 02_check_dataset.py")