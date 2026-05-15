

import os
import sys
from pathlib import Path

class ImageRenamer:
    
    def __init__(self):
        self.total_renamed = 0
        self.total_folders = 0
        self.total_skipped = 0
    
    def rename_images_in_folder(self, folder_path, prefix):
       
        
        if not os.path.exists(folder_path):
            print(f"⚠️  Folder not found: {folder_path}")
            return 0
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        files = [f for f in os.listdir(folder_path) if f.endswith(valid_extensions)]
        
        if len(files) == 0:
            print(f"⏭️  Skipped (empty): {folder_path}")
            self.total_skipped += 1
            return 0
        
        files.sort()
        
        print(f"\n📁 Processing: {os.path.basename(folder_path)}")
        print(f"   Found {len(files)} images")
        print(f"   Prefix: {prefix}")
        
        renamed_count = 0
        
        for idx, old_filename in enumerate(files, 1):
            try:
                file_ext = os.path.splitext(old_filename)[1]
                
                new_filename = f"{prefix}_{idx:03d}{file_ext}"
                
                old_path = os.path.join(folder_path, old_filename)
                new_path = os.path.join(folder_path, new_filename)
                
                os.rename(old_path, new_path)
                
                if idx % 10 == 0 or idx == len(files):
                    print(f"   ✓ {idx}/{len(files)} renamed")
                
                renamed_count += 1
            
            except Exception as e:
                print(f"   ✗ Error renaming {old_filename}: {e}")
        
        return renamed_count
    
    def run(self):
        
        print("\n" + "=" * 80)
        print("IMAGE BATCH RENAME SCRIPT")
        print("=" * 80)
        print("\nThis script renames images to standard naming convention:")
        print("  Format: {category}_{sequence_number}.jpg")
        print("  Example: no_mask_001.jpg, surgical_050.jpg, white_100.jpg")
        
        folder_mappings = {
            'raw_data/status/no_mask': 'no_mask',
            'raw_data/status/proper_mask': 'proper_mask',
            'raw_data/status/improper_mask': 'improper_mask',
            
            'raw_data/type/surgical': 'surgical',
            'raw_data/type/n95': 'n95',
            'raw_data/type/cloth': 'cloth',
            
            'raw_data/color/white': 'white',
            'raw_data/color/black': 'black',
            'raw_data/color/blue': 'blue',
        }
        
        print(f"\n{'='*80}")
        print("STARTING BATCH RENAME")
        print(f"{'='*80}\n")
        
        for folder_path, prefix in folder_mappings.items():
            if os.path.exists(folder_path):
                count = self.rename_images_in_folder(folder_path, prefix)
                self.total_renamed += count
                self.total_folders += 1
            else:
                print(f"⚠️  Folder not found: {folder_path}")
        
        print("\n" + "=" * 80)
        print("RENAME COMPLETE - SUMMARY")
        print("=" * 80)
        print(f"\n✅ Statistics:")
        print(f"   Folders processed: {self.total_folders}")
        print(f"   Folders skipped (empty): {self.total_skipped}")
        print(f"   Total images renamed: {self.total_renamed}")
        print(f"\n✓ All images have been renamed to standard convention!")
        print(f"   Format: {{category}}_{{number}}.jpg")
        print(f"\nExample results:")
        print(f"   no_mask_001.jpg, no_mask_002.jpg, ..., no_mask_100.jpg")
        print(f"   surgical_001.jpg, surgical_002.jpg, ..., surgical_100.jpg")
        print(f"   white_001.jpg, white_002.jpg, ..., white_100.jpg")
        print("=" * 80 + "\n")
        
        return self.total_renamed > 0

if __name__ == "__main__":
    renamer = ImageRenamer()
    success = renamer.run()
    
    if success:
        print("💾 Ready to proceed with preprocessing!")
    else:
        print("⚠️  No images were renamed. Check your folder structure.")