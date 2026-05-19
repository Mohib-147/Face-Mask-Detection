import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.config import MODEL1_PATH, MODEL2_PATH, MODEL3_PATH
from scripts.utils.inference import ImagePreprocessor, ModelLoader

def print_banner():
    print("\n" + "="*80)
    print("FACE MASK DETECTION - MAIN INTERFACE")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Face Mask Detection System')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('model', type=int, choices=[1, 2, 3], help='Model to train (1, 2, or 3)')
    
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('image', type=str, help='Path to image file')
    
    infer_parser = subparsers.add_parser('infer', help='Real-time inference')
    infer_parser.add_argument('--source', type=str, default='webcam', help='Input source (webcam or video file)')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('model', type=int, choices=[1, 2, 3, 0], help='Model to evaluate (0 for all)')
    
    setup_parser = subparsers.add_parser('setup', help='Setup project')
    
    check_parser = subparsers.add_parser('check', help='Check dataset')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == 'train':
        from scripts import evaluate as train_module
        if args.model == 1:
            os.system('python scripts/04_train_model1_status.py')
        elif args.model == 2:
            os.system('python scripts/05_train_model2_colour.py')
        elif args.model == 3:
            os.system('python scripts/06_train_model3_type.py')
    
    elif args.command == 'predict':
        from scripts.utils.inference import ImagePreprocessor
        os.system(f'python scripts/07_predict.py {args.image}')
    
    elif args.command == 'infer':
        if args.source == 'webcam':
            os.system('python scripts/08_inference.py webcam')
        else:
            os.system(f'python scripts/08_inference.py {args.source}')
    
    elif args.command == 'evaluate':
        if args.model == 0:
            os.system('python scripts/09_evaluate_all_models.py')
        else:
            print(f"Evaluating Model {args.model}...")
            os.system(f'python scripts/09_evaluate_all_models.py {args.model}')
    
    elif args.command == 'setup':
        os.system('python scripts/01_setup_folders.py')
    
    elif args.command == 'check':
        os.system('python scripts/02_check_dataset.py')
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
