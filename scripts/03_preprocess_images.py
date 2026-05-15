import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import sys

class ImagePreprocessor:
    
    TARGET_SIZE = (224, 224)  
    
    def __init__(self):
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
    
    def preprocess_single_image(self, image_path):
        
        try:
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"     Failed to read: {os.path.basename(image_path)}")
                self.failed_count += 1
                return None
            
            resized = cv2.resize(img, self.TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            
            normalized = resized.astype('float32') / 255.0
            
            return normalized
        
        except Exception as e:
            print(f"    Error processing {os.path.basename(image_path)}: {e}")
            self.failed_count += 1
            return None
    
    def save_preprocessed_image(self, image_array, output_path):
        try:
            image_uint8 = (image_array * 255).astype('uint8')
            cv2.imwrite(output_path, image_uint8)
            return True
        except Exception as e:
            print(f"     Error saving to {output_path}: {e}")
            return False
    
    def process_category_folder(self, input_folder, output_folder, category_name):
        
        if not os.path.exists(input_folder):
            print(f"  Input folder not found: {input_folder}")
            return 0
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        image_files = [f for f in os.listdir(input_folder) if f.endswith(valid_extensions)]
        
        if len(image_files) == 0:
            print(f"   No images found in {category_name}")
            return 0
        
        print(f"  Processing {category_name}: {len(image_files)} images")
        
        os.makedirs(output_folder, exist_ok=True)
        
        processed = 0
        
        for idx, filename in enumerate(image_files):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            preprocessed = self.preprocess_single_image(input_path)
            
            if preprocessed is not None:
                if self.save_preprocessed_image(preprocessed, output_path):
                    processed += 1
                    self.processed_count += 1
            
            if (idx + 1) % 25 == 0 or (idx + 1) == len(image_files):
                print(f"     {idx + 1}/{len(image_files)} processed")
        
        print(f"  Completed {category_name}: {processed}/{len(image_files)} images saved\n")
        return processed

def preprocess_all_datasets():
    
    print("=" * 80)
    print("IMAGE PREPROCESSING")
    print("=" * 80)
    print(f"\nTarget size: {ImagePreprocessor.TARGET_SIZE}")
    print("Normalization: Pixel values 0-255 → 0-1")
    print("\n" + "-" * 80)
    
    preprocessor = ImagePreprocessor()
    
    datasets = {
        'MODEL 1 - MASK STATUS': {
            'raw_path': 'raw_data/status',
            'processed_path': 'dataset/model1_mask_status',
            'categories': ['no_mask', 'proper_mask', 'improper_mask']
        },
        'MODEL 2 - MASK TYPE': {
            'raw_path': 'raw_data/type',
            'processed_path': 'dataset/model2_mask_type',
            'categories': ['surgical', 'n95', 'cloth']
        },
        'MODEL 3 - MASK COLOR': {
            'raw_path': 'raw_data/color',
            'processed_path': 'dataset/model3_mask_color',
            'categories': ['white', 'black', 'blue']
        }
    }
    
    total_processed = 0
    
    for dataset_name, config in datasets.items():
        print(f"\n {dataset_name}")
        print("-" * 80)
        
        dataset_total = 0
        
        for category in config['categories']:
            input_folder = os.path.join(config['raw_path'], category)
            output_folder = os.path.join(config['processed_path'], category)
            
            processed = preprocessor.process_category_folder(
                input_folder, 
                output_folder, 
                category
            )
            dataset_total += processed
            total_processed += processed
        
        print(f"Dataset Total: {dataset_total} images processed\n")
    
    print("=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f" Successfully processed: {preprocessor.processed_count} images")
    print(f"  Failed: {preprocessor.failed_count} images")
    print(f"Total: {preprocessor.processed_count + preprocessor.failed_count} images")
    print("=" * 80)
    
    return preprocessor.processed_count > 0

def split_train_test(preprocessor):
    
    
    print("\n" + "=" * 80)
    print("CREATING TRAIN/TEST SPLIT (80/20)")
    print("=" * 80)
    
    datasets = {
        'MODEL 1 - MASK STATUS': {
            'processed_path': 'dataset/model1_mask_status',
            'categories': ['no_mask', 'proper_mask', 'improper_mask']
        },
        'MODEL 2 - MASK TYPE': {
            'processed_path': 'dataset/model2_mask_type',
            'categories': ['surgical', 'n95', 'cloth']
        },
        'MODEL 3 - MASK COLOR': {
            'processed_path': 'dataset/model3_mask_color',
            'categories': ['white', 'black', 'blue']
        }
    }
    
    total_train = 0
    total_test = 0
    
    for dataset_name, config in datasets.items():
        print(f"\n{dataset_name}")
        print("-" * 80)
        
        for category in config['categories']:
            source_folder = os.path.join(config['processed_path'], category)
            
            train_folder = os.path.join(config['processed_path'], 'train', category)
            test_folder = os.path.join(config['processed_path'], 'test', category)
            
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)
            
            valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
            images = [f for f in os.listdir(source_folder) if f.endswith(valid_extensions)]
            
            if len(images) == 0:
                print(f"    No images in {category}")
                continue
            
            train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
            
            for img in train_images:
                src = os.path.join(source_folder, img)
                dst = os.path.join(train_folder, img)
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"    Error copying train image {img}: {e}")
            
            for img in test_images:
                src = os.path.join(source_folder, img)
                dst = os.path.join(test_folder, img)
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"   Error copying test image {img}: {e}")
            
            print(f"  {category:20s}: Train {len(train_images):3d} | Test {len(test_images):3d}")
            
            total_train += len(train_images)
            total_test += len(test_images)
        
        print()
    
    print("=" * 80)
    print("TRAIN/TEST SPLIT SUMMARY")
    print("=" * 80)
    print(f" Training images:   {total_train} (80%)")
    print(f"Test images:       {total_test} (20%)")
    print(f"Total:              {total_train + total_test} images")
    print("=" * 80)

if __name__ == "__main__":
    print("\n")
    
    try:
        success = preprocess_all_datasets()
        
        if not success:
            print("\n Preprocessing failed!")
            sys.exit(1)
        
        preprocessor = ImagePreprocessor()
        split_train_test(preprocessor)
        
        print("\n" + "=" * 80)
        print(" PREPROCESSING COMPLETE!")
        print("=" * 80)
        print("\nFolder structure created:")
        print("  dataset/model1_mask_status/train/  ← Training images for Model 1")
        print("  dataset/model1_mask_status/test/   ← Test images for Model 1")
        print("  dataset/model2_mask_type/train/    ← Training images for Model 2")
        print("  dataset/model2_mask_type/test/     ← Test images for Model 2")
        print("  dataset/model3_mask_color/train/   ← Training images for Model 3")
        print("  dataset/model3_mask_color/test/    ← Test images for Model 3")
        print("\n Next step: Run 04_split_train_test.py")
        print("   OR directly run: 05_train_model1_status.py")
        print("=" * 80)
        print("\n")
        
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        sys.exit(1)
