import sys
import argparse
import warnings

from model import FeatureFusion3D, LearnedFeatureFusion3D
from dataset import *
from utils import *
from utils import get_date

from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR


def main(args):
    """Train model and evaluate on test set."""
    print(args)

    # Set random seeds for reproducibility
    set_seed(args.seed)

    # Set device for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Get train and val data
    train_data = HipFusionDataset(img_path=args.train_img_path, label_path=args.train_label_path, augment=args.augment, sheet_name="2year model")
    val_data   = HipFusionDataset(img_path=args.validation_img_path, label_path=args.validation_label_path, augment=False, sheet_name="2 year FU model")
    test_data  = HipFusionDataset(img_path=args.test_img_path, label_path=args.test_label_path, augment=False, sheet_name="2 year FU model")

    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=8)
    val_loader   = DataLoader(val_data, shuffle=False, batch_size=1, num_workers=4)
    test_loader  = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=4)

    # Define model
    # if args.model == "image-only":
    #     model = ResNet50(pre_trained=args.pretrained, frozen=False).to(device)
    #     fusion, meta_only = False, False
    # elif args.model == "non-image-only":
    #     model = ShallowFFNN(meta_features=train_data.meta_features).to(device)
    #     fusion, meta_only = False, True
    if args.model == "feature-fusion":
        model = FeatureFusion3D(meta_features=train_data.meta_features, pre_trained=False, frozen=False).to(device)
        fusion, meta_only = True, False
    elif args.model == "learned-feature-fusion":
        # if args.train_mode == "default":
        model = LearnedFeatureFusion3D(meta_features=train_data.meta_features, mode=args.fusion_mode, pre_trained=False, frozen=False).to(device)
    #     elif args.train_mode == "multiloss" or args.train_mode == "multiopt":
    #         model = LearnedFeatureFusionVariant(meta_features=train_data.meta_features, mode=args.fusion_mode, pre_trained=args.pretrained, frozen=False).to(device)
    #     else:
    #         sys.exit("Invalid train_mode specified")
    #     fusion, meta_only = True, False
    # elif args.model == "probability-fusion":
    #     model = ProbabilityFusion(meta_features=train_data.meta_features, pre_trained=args.pretrained, frozen=False).to(device)
    #     fusion, meta_only = True, False
    else:
        sys.exit("Invalid model specified.")

    # Choose proper train and evaluation functions based on optimization approach
    if args.train_mode == "default":
        train_fxn = train
        eval_fxn = evaluate
    # elif args.train_mode == "multiloss":
    #     train_fxn = train_multiloss
    #     eval_fxn = evaluate_multiloss
    # elif args.train_mode == "multiopt":
    #     train_fxn = train_multiopt
    #     eval_fxn = evaluate
    lr = args.lr
    if args.optimizer == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=args.lr
        )
    elif args.optimizer == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=lr
        )
    elif args.optimizer == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=lr
        )
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(
            model.parameters(),
            lr=lr
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=args.min_lr
    )
    
    model, history = train_fxn(model, train_loader, args.max_epochs, optimizer, device, val_loader, fusion=True, meta_only=False, lr_scheduler=lr_scheduler, label_smoothing=args.label_smoothing)
    
    MODEL_NAME = f"{get_date()}_{args.model}"
    result = eval_fxn(model, test_loader, device, fusion=True, meta_only=False)
    torch.save(model.state_dict(), os.path.join(args.out_dir, MODEL_NAME + ".pt"))
    
if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_img_path", default="./2year_training", type=str,
                        help="path to image")
    parser.add_argument("--validation_img_path", default="./2year_validation", type=str,
                        help="path to image")
    parser.add_argument("--test_img_path", default="./2year_testing", type=str,
                        help="path to image")
    parser.add_argument("--train_label_path", default="./Classification_THP_1301_modified.xlsx", type=str,
                        help="path to label file")
    parser.add_argument("--validation_label_path", default="./External_valdidation_THP1401_modified.xlsx", type=str,
                        help="path to label file")
    parser.add_argument("--test_label_path", default="./External_valdidation_THP1401_modified.xlsx", type=str,
                        help="path to label file")
    parser.add_argument("--out_dir", default="./", type=str,
                        help="path to directory where results and model weights will be saved")        
    parser.add_argument("--model", default="learned-feature-fusion", type=str,
                        help="must be one of ['image-only', 'shallow-only', 'feature-fusion', 'hidden-feature-fusion', 'probability-fusion', 'learned-feature-fusion']")
    parser.add_argument("--train_mode", default="default", type=str,
                        help="approach to optimizing fusion model (one of ['default', 'multiloss', 'multiopt']")
    parser.add_argument("--fusion_mode", default="concat", help="fusion type for LearnedFeatureFusion or ProbabilityFusion (one of ['concat', 'multiply', 'add'])")
    parser.add_argument("--max_epochs", default=100, type=int, help="maximum number of epochs to train")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for training, validation, and testing")
    parser.add_argument("--augment", default=False, action="store_true", help="whether or not to use augmentation during training")
    parser.add_argument("--seed", default=42, type=int, help="set random seed")
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")
    parser.add_argument("--min_lr", default=1e-7, type=float, help="min initial learning rate")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw', 'rmsprop'],
                        help='Optimizer to use (default: adam)')
    parser.add_argument("--label_smoothing", default=0.01, type=float, help="ratio of label smoothing to use during training")
    

    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    main(args)