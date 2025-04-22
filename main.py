import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='MissingModality - A deep learning framework for missing modality training/testing')
    parser.add_argument('action', type=str, choices=['train', 'test'], 
                        help='Action to perform: train or test')
    
    # Add all other arguments to be parsed by the specific scripts
    args, unknown_args = parser.parse_known_args()
    
    return args, unknown_args

def main():
    args, unknown_args = parse_args()
    
    # Prepare the command for the specific script
    if args.action == 'train':
        # Import and run the train script
        from train import main as train_main
        sys.argv = [sys.argv[0]] + unknown_args
        train_main()
    elif args.action == 'test':
        # Import and run the test script
        from test import main as test_main
        sys.argv = [sys.argv[0]] + unknown_args
        test_main()
    else:
        print(f"Unknown action: {args.action}")
        sys.exit(1)

if __name__ == "__main__":
    main()