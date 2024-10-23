import argparse
import os

def validate_path(path):
    return isinstance(path, str) and os.path.exists(path)

def validate_mode(mode):
    return isinstance(mode, str) and mode.lower() in ["2d", "3d"]

def validate_imp(imp):
    return imp == None or validate_path(imp)

def validate_epochs(epochs):
    return isinstance(epochs, int) and epochs > 0

def validate_train_args(args):
    if not validate_mode(args.mode):
        raise ValueError(f"Incorrect mode argument: {args.mode}; expected 2d or 3d")
    
    if not validate_imp(args.input_model_path):
        raise ValueError(f"Incorrect imp argument: {args.input_model_path}; expected None or a valid path")
        
    if not validate_path(args.train_ds_path):
        raise ValueError(f"Incorrect train-ds-path argument: {args.train_ds_path}; expected a valid path")
    
    if not validate_path(args.val_ds_path):
        raise ValueError(f"Incorrect val-ds-path argument: {args.val_ds_path}; expected a valid path")
    
    if not validate_epochs(args.epochs):
        raise ValueError(f"Incorrect value for epochs provided: {args.epochs}; Expected int > 0")   

def validate_test_args(args):
    if not validate_mode(args.mode):
        raise ValueError(f"Incorrect mode argument: {args.mode}; expected 2d or 3d")
                
    if not validate_path(args.val_ds_path):
        raise ValueError(f"Incorrect val-ds-path argument: {args.val_ds_path}; expected a valid path")
    
def get_train_args():
    parser = argparse.ArgumentParser(description="Arguments for train script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--mode", help="Mode 2D or 3D", default="3d")
    parser.add_argument("-imp", "--input-model-path", help="input model path", default=None)
    parser.add_argument("-omp", "--output-model-path", help="output model path", default="models/model.pth")
    parser.add_argument("--is-regression", action="store_true", help="is regression or segmentation", default=False)
    parser.add_argument("--is-mip", action="store_true", help="use Max Intensity Projection (MIP)", default=False)
    
    parser.add_argument("--train-ds-path", default="/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2", help="path to the training images")
    parser.add_argument("--val-ds-path", default="/home/mali2/datasets/CellSeg/Widefield Deconvolved", help="path to the val images")

    parser.add_argument("-e", "--epochs", default=500, type=int, help="Number of epochs to train the model")

    args = parser.parse_args()
    validate_train_args(args)
    
    return args

def get_test_args():
    parser = argparse.ArgumentParser(description="Arguments for the test script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--mode", help="Mode 2D or 3D", default="3d")
    parser.add_argument("-imp", "--input-model-path", help="input model path")
    parser.add_argument("--is-regression", action="store_true", help="is regression or segmentation", default=False)
    parser.add_argument("--is-mip", action="store_true", help="use Max Intensity Projection (MIP)", default=False)

    parser.add_argument("--val-ds-path", default="/home/mali2/datasets/CellSeg/Widefield Deconvolved", help="path to the val images")
    parser.add_argument("--output-path", default="/home/mali2/datasets/CellSeg/generated", help="output path for storing inference output")

    args = parser.parse_args()
    validate_test_args(args)
    
    return args

def get_mode(args) -> str:
    return args.mode.lower()

def get_input_model_path(args) -> str|None:
    return args.input_model_path

def get_output_model_path(args) -> str:
    return args.output_model_path

def get_train_ds_path(args) -> str:
    return args.train_ds_path

def get_val_ds_path(args) -> str:
    return args.val_ds_path

def get_output_path(args) -> str:
    return args.output_path

def get_num_epochs(args) -> int:
    return int(args.epochs)

def is_segmentation(args) -> bool:
    return not args.is_regression

def is_mip(args) -> bool:
    return args.is_mip