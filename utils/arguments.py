import argparse
import os

def validate_path(path):
    return isinstance(path, str) and os.path.exists(path)

def validate_mode(mode):
    return isinstance(mode, str) and mode.lower() not in ["2d", "3d"]

def validate_imp(imp):
    return imp == None or validate_path(imp)

def validate_train_args(args):
    if not validate_mode(args.mode):
        raise ValueError(f"Incorrect mode argument: {args.mode}; expected 2d or 3d")
    
    if not validate_imp(args.imp):
        raise ValueError(f"Incorrect imp argument: {args.imp}; expected None or a valid path")
    
    if not validate_path(args.omp):
        raise ValueError(f"Incorrect omp argument: {args.omp}; expected a valid path")
    
    if not validate_path(args.train_ds_path):
        raise ValueError(f"Incorrect train-ds-path argument: {args.train_ds_path}; expected a valid path")
    
    if not validate_path(args.val_ds_path):
        raise ValueError(f"Incorrect val-ds-path argument: {args.val_ds_path}; expected a valid path")
    
    if not validate_path(args.output_path):
        raise ValueError(f"Incorrect output-path argument: {args.output_path}; expected a valid path")      

def validate_test_args(args):
    if not validate_mode(args.mode):
        raise ValueError(f"Incorrect mode argument: {args.mode}; expected 2d or 3d")
    
    if not validate_path(args.imp):
        raise ValueError(f"Incorrect imp argument: {args.imp}; expected None or a valid path")
            
    if not validate_path(args.val_ds_path):
        raise ValueError(f"Incorrect val-ds-path argument: {args.val_ds_path}; expected a valid path")
    
    if not validate_path(args.output_path):
        raise ValueError(f"Incorrect output-path argument: {args.output_path}; expected a valid path")        

def get_train_args():
    parser = argparse.ArgumentParser(description="Arguments for train script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--mode", help="Mode 2D or 3D", default="3d")
    parser.add_argument("-imp", "--input-model-path", help="input model path", default=None)
    parser.add_argument("-omp", "--ouput-model-path", help="output model path", default="models/model.pth")
    parser.add_argument("--is-regression", action="store_true", help="is regression or segmentation", default=False)
    parser.add_argument("--is-mip", action="store_true", help="use Max Intensity Projection (MIP)", default=False)
    
    parser.add_argument("--train-ds-path", default="/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2", help="path to the training images")
    parser.add_argument("--val-ds-path", default="/home/mali2/datasets/CellSeg/Widefield Deconvolved", help="path to the val images")
    parser.add_argument("--output-path", default="/home/mali2/datasets/CellSeg/generated", help="output path for storing inference output")

    args = parser.parse_args()
    return vars(args)

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
    return vars(args)

def get_mode(args) -> str:
    return args.mode.lower()

def get_input_model_path(args) -> str|None:
    return args.imp

def get_output_model_path(args) -> str:
    return args.omp

def get_train_ds_path(args) -> str:
    return args.train_ds_path

def get_val_ds_path(args) -> str:
    return args.val_ds_path

def get_output_path(args) -> str:
    return args.output_path

def is_segmentation(args) -> bool:
    return not args.is_regression

def is_mip(args) -> bool:
    return args.is_mip