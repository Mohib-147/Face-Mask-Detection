import os
import sys

def count_images_in_folder(folder_path):
    """Count valid image files in a folder"""
    if not os.path.exists(folder_path):
        return 0
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    images = [f for f in os.listdir(folder_path) if f.endswith(valid_extensions)]
    
    return len(images)

def check_raw_data():
    """Check all raw_data folders and report counts"""
    
    print("=" * 80)
    print("CHECKING DATASET (100 images per label requirement)")
    print("=" * 80)
    
    structure = {
        'MASK STATUS (3 labels)': {
            'path': 'raw_data/status',
            'categories': ['no_mask', 'proper_mask', 'improper_mask'],
            'target_per_category': 100,  
            'total_labels': 3
        },
        'MASK TYPE (3 labels)': {
            'path': 'raw_data/type',
            'categories': ['surgical', 'n95', 'cloth'],
            'target_per_category': 100,  
            'total_labels': 3
        },
        'MASK COLOR (3 labels)': {
            'path': 'raw_data/color',
            'categories': ['white', 'black', 'blue'],
            'target_per_category': 100,  
            'total_labels': 3
        }
    }
    
    total_images = 0
    total_labels = 0
    all_complete = True
    
    for dataset_name, config in structure.items():
        print(f"\n📂 {dataset_name}")
        print("-" * 80)
        
        dataset_path = config['path']
        categories = config['categories']
        target = config['target_per_category']
        
        if not os.path.exists(dataset_path):
            print(f"❌ Folder not found: {dataset_path}")
            all_complete = False
            continue
        
        category_total = 0
        category_complete = 0
        
        for category in categories:
            category_path = os.path.join(dataset_path, category)
            count = count_images_in_folder(category_path)
            category_total += count
            total_images += count
            total_labels += 1
            
            # Check status - UPDATED for 100 images requirement
            if count == 0:
                status = "❌ EMPTY - Add images"
                all_complete = False
            elif count < target * 0.5:  # Less than 50 images
                status = f"⚠️  INCOMPLETE ({count}/100)"
                all_complete = False
            elif count < target:  # 50-99 images
                status = f"⚠️  ALMOST THERE ({count}/100)"
                all_complete = False
            elif count == target:  # Exactly 100
                status = f"✅ COMPLETE ({count}/100)"
                category_complete += 1
            else:  # More than 100
                status = f"✅ COMPLETE+ ({count}/100) - Extra images: {count - target}"
                category_complete += 1
            
            print(f"  {category:20s}: {status}")
        
        print(f"  {'-'*76}")
        print(f"  Subtotal: {category_total}/300 images | Complete: {category_complete}/3 labels")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    required_total = 900  # 9 labels × 100 images
    
    print(f"\nTotal Images Collected: {total_images}/900")
    print(f"Images per Label (Average): {total_images / total_labels:.1f}/100")
    
    # Status for each dataset
    print("\n" + "-" * 80)
    print("STATUS BY DATASET:")
    print("-" * 80)
    
    status_count = check_individual_dataset('raw_data/status', ['no_mask', 'proper_mask', 'improper_mask'], 100)
    type_count = check_individual_dataset('raw_data/type', ['surgical', 'n95', 'cloth'], 100)
    color_count = check_individual_dataset('raw_data/color', ['white', 'black', 'blue'], 100)
    
    print(f"Status Labels Complete:  {status_count}/3 ({'✅' if status_count == 3 else '⚠️ '})")
    print(f"Type Labels Complete:    {type_count}/3 ({'✅' if type_count == 3 else '⚠️ '})")
    print(f"Color Labels Complete:   {color_count}/3 ({'✅' if color_count == 3 else '⚠️ '})")
    
    # Final Verdict
    print("\n" + "=" * 80)
    
    if total_images == required_total:
        print("✅ PERFECT! All 900 images collected (100 per label)")
        print("   Ready to proceed with preprocessing!")
        return True
    elif total_images >= 800:
        print(f"✅ GOOD! {total_images}/900 images collected")
        print("   You can proceed, but consider adding more for better results")
        return True
    elif total_images >= 600:
        print(f"⚠️  ACCEPTABLE ({total_images}/900 images)")
        print("   Recommend adding more images for better model performance")
        return True
    else:
        print(f"❌ INSUFFICIENT ({total_images}/900 images)")
        print(f"   Please collect at least {900 - total_images} more images")
        return False

def check_individual_dataset(path, categories, target):
    """Check how many categories have reached target"""
    if not os.path.exists(path):
        return 0
    
    complete_count = 0
    for category in categories:
        category_path = os.path.join(path, category)
        count = count_images_in_folder(category_path)
        if count >= target:
            complete_count += 1
    
    return complete_count

def get_detailed_breakdown():
    """Show what's missing"""
    
    print("\n" + "=" * 80)
    print("WHAT'S MISSING (if any)")
    print("=" * 80)
    
    structure = {
        'raw_data/status': ['no_mask', 'proper_mask', 'improper_mask'],
        'raw_data/type': ['surgical', 'n95', 'cloth'],
        'raw_data/color': ['white', 'black', 'blue']
    }
    
    total_missing = 0
    
    for dataset_path, categories in structure.items():
        if not os.path.exists(dataset_path):
            continue
        
        for category in categories:
            category_path = os.path.join(dataset_path, category)
            count = count_images_in_folder(category_path)
            missing = max(0, 100 - count)
            
            if missing > 0:
                print(f"  {category:20s}: Need {missing:3d} more images (have {count}/100)")
                total_missing += missing
    
    if total_missing == 0:
        print("  ✅ Nothing missing! All labels complete.")
    else:
        print(f"\n  Total missing: {total_missing} images")
    
    return total_missing

def show_collection_guide():
    """Show user what to collect"""
    
    print("\n" + "=" * 80)
    print("IMAGE COLLECTION GUIDE")
    print("=" * 80)
    
    print("""
You need to collect 900 images total (100 per label):

MASK STATUS LABELS (collect 300 images):
  ├─ no_mask/           (100 images) - People without masks
  ├─ proper_mask/       (100 images) - Masks worn correctly (nose + mouth covered)
  └─ improper_mask/     (100 images) - Masks worn incorrectly

MASK TYPE LABELS (collect 300 images from masked people):
  ├─ surgical/          (100 images) - Medical/surgical masks
  ├─ n95/               (100 images) - N95/KN95 respirators
  └─ cloth/             (100 images) - Cloth masks

MASK COLOR LABELS (collect 300 images from masked people):
  ├─ white/             (100 images) - White/light colored masks
  ├─ black/             (100 images) - Black masks
  └─ blue/              (100 images) - Blue masks

💡 TIP: You can take photos yourself, screenshot from videos, 
   or download from free image search engines (Google Images, Bing, etc.)
""")

if __name__ == "__main__":
    print("\n")
    
    # Show collection guide
    show_collection_guide()
    
    # Check dataset
    is_ready = check_raw_data()
    
    # Show what's missing
    missing = get_detailed_breakdown()
    
    print("\n" + "=" * 80)
    if is_ready and missing == 0:
        print("✅ READY TO PROCEED")
        print("   Next step: Run 03_preprocess_images.py")
    elif is_ready and missing > 0:
        print(f"⚠️  ALMOST READY ({900 - missing}/900 images)")
        print(f"   Collect {missing} more images, then run 03_preprocess_images.py")
    else:
        print("❌ NOT READY YET")
        print(f"   Collect {missing} more images first")
    print("=" * 80)
    print("\n")