import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import pandas as pd
import random
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


writer = SummaryWriter('runs/experiment_1')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_date():
    now = datetime.now()

    if now.month < 10:
        str_month = "0" + str(now.month)
    else:
        str_month = str(now.month)

    if now.day < 10:
        str_day = "0" + str(now.day)
    else:
        str_day = str(now.day)

    str_year = str(now.year)[2:]

    return str_month + str_day + str_year

class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation for handling class imbalance problems
    
    Args:
        alpha: Class weight, can be a scalar (for positive samples) or tensor (weight for each class)
        gamma: Modulation factor, increases focus on hard-to-classify examples
        reduction: 'none' | 'mean' | 'sum'
        label_smoothing: Label smoothing parameter, 0 means no label smoothing
        dynamic_weight: Whether to calculate class weights dynamically based on batch distribution
    """
    def __init__(self, alpha=3.0, gamma=4.0, reduction='mean', label_smoothing=0.1, dynamic_weight=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.dynamic_weight = dynamic_weight
        
    def forward(self, inputs, targets):
        """
        Calculate Focal Loss
        
        Args:
            inputs: Model predictions (logits), shape [batch_size, 1]
            targets: Ground truth labels, shape [batch_size, 1]
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth_targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smooth_targets = targets
            
        # Calculate dynamic class weights
        if self.dynamic_weight:
            # Calculate class distribution in current batch
            pos_weight = (1 - targets.mean()) / (targets.mean() + 1e-6)
            # Limit weights to a reasonable range
            pos_weight = torch.clamp(pos_weight, min=1.0, max=5.0)
        else:
            pos_weight = torch.tensor(self.alpha, device=inputs.device)
        
        # Calculate initial loss using weighted BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, 
            smooth_targets, 
            reduction='none',
            pos_weight=pos_weight
        )
        
        # Get prediction probabilities
        probs = torch.sigmoid(inputs)
        # Adjust probabilities based on targets (use p for target=1, 1-p for target=0)
        pt = torch.where(smooth_targets > 0.5, probs, 1 - probs)
        
        # Calculate Focal Loss modulation factor
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight to BCE loss
        loss = focal_weight * bce_loss
        
        # Return result based on reduction mode
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # 'sum'
            return loss.sum()
        
def train(model, train_loader, max_epochs, optim, device, val_loader, fusion=True, 
         meta_only=False, lr_scheduler=None, label_smoothing=0.0, focal_gamma=4.0, dynamic_weight=True):
    """
    Train a model with Focal Loss and TensorBoard logging.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader containing training data
        max_epochs: Maximum number of epochs to train for
        optim: Optimizer for parameter updates
        device: Device to run computations on (CPU/GPU)
        val_loader: DataLoader containing validation data (optional)
        fusion: If True, use multimodal fusion with images and metadata
        meta_only: If True, use only metadata for prediction
        lr_scheduler: Cosine decay learning rate scheduler
        label_smoothing: Label smoothing parameter
        focal_gamma: Gamma parameter for Focal Loss (higher values increase focus on hard examples)
        dynamic_weight: Whether to calculate class weights dynamically
        
    Returns:
        model: Trained model with best weights according to validation accuracy
        history: DataFrame containing training metrics history
    """
    # Dictionary to store training and validation metrics for each epoch
    history = {"epoch": [], "loss": [], "auc_roc": [], "auc_pr": [], "acc": [], "precision": [], "recall": [], "f1": [],
               "val_loss": [], "val_auc_roc": [], "val_auc_pr": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": [], "lr": []}
    
    # Initialize Focal Loss
    loss_fxn = FocalLoss(
        alpha=3.0,
        gamma=focal_gamma,
        reduction='mean',
        label_smoothing=label_smoothing,
        dynamic_weight=dynamic_weight
    ).to(device)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter()
    
    # Variables to track best model
    best_val_acc = 0.0
    best_model_state = None
    
    # Main training loop
    for epoch in range(1, max_epochs + 1):
        # Set model to training mode
        model.train()

        # Create progress bar for current epoch
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        pbar.set_description_str(f"Epoch {epoch}")

        # Initialize metrics for current epoch
        n = 0  # Total samples processed
        tn = 0  # True negatives
        fp = 0  # False positives
        fn = 0  # False negatives
        tp = 0  # True positives
        running_loss = 0.  # Cumulative loss
        
        # Iterate through batches
        for step, batch in pbar:
            # Zero gradients before forward pass
            optim.zero_grad()

            # Move batch data to device
            x = batch["image"].to(device)
            meta = batch["metadata"].to(device)
            y = batch["label"].to(device)

            # Forward pass based on selected mode (fusion/meta_only/image_only)
            if fusion:
                y_hat = model.forward(x, meta)  # Multimodal fusion
            elif meta_only:
                y_hat = model.forward(meta)  # Metadata only
            else:
                y_hat = model.forward(x)  # Image only
            
            # Binary predictions
            y_pred = torch.round(torch.sigmoid(y_hat))

            # Collect predictions and ground truth for AUC calculation
            if step == 0:
                y_prob = torch.sigmoid(y_hat).cpu().detach()
                y_true = y.cpu().detach()
            else:
                y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

            # Calculate loss with focal loss
            loss = loss_fxn(y_hat, y.view(-1, 1))
                
            # Backward pass and optimization
            loss.backward()
            optim.step()  # Update model parameters
            
            # Update running metrics
            running_loss += loss.item() * y.shape[0]
            n += y.shape[0]
            
            # Calculate confusion matrix elements
            for i in range(y.shape[0]):
                if y[i] == 1 and y_pred[i] == 1:
                    tp += 1
                if y[i] == 1 and y_pred[i] == 0:
                    fn += 1
                if y[i] == 0 and y_pred[i] == 1:
                    fp += 1
                if y[i] == 0 and y_pred[i] == 0:
                    tn += 1

            # Calculate current metrics
            l = running_loss / n  # Average loss
            a = (tp + tn) / n  # Accuracy
            
            # Precision (positive predictive value)
            if tp + fp > 0:
                pr = tp / (tp + fp)
            else:
                pr = np.nan
                
            # Recall/Sensitivity (true positive rate)    
            if tp + fn > 0:
                re = tp / (tp + fn)
            else:
                re = np.nan
                
            # F1 score
            if tp + fp + fn > 0:
                f1 = 2 * tp / (2 * tp + fp + fn)
            else:
                f1 = np.nan

            # Update progress bar with current metrics
            pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "f1": f1})
            pbar.refresh()
            time.sleep(0.01)

        # Calculate AUC metrics on entire training set
        # y_prob = np.nan_to_num(y_prob, nan=0.5)  # Replace NaNs with 0.5
        auc_roc = roc_auc_score(y_true, y_prob)  # Area under ROC curve
        prs, res, thrs = precision_recall_curve(y_true, y_prob)
        auc_pr = auc(res, prs)  # Area under Precision-Recall curve
        print("\tAUC ROC:", round(auc_roc, 3), "| AUC PR:", round(auc_pr, 3))

        # Store training metrics in history
        history["loss"].append(l)
        history["auc_roc"].append(auc_roc)
        history["auc_pr"].append(auc_pr)
        history["acc"].append(a)
        history["precision"].append(pr)
        history["recall"].append(re)
        history["f1"].append(f1)
        
        # Record current learning rate
        current_lr = optim.param_groups[0]['lr']
        history["lr"].append(current_lr)

        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', l, epoch)
        writer.add_scalar('Accuracy/train', a, epoch)
        writer.add_scalar('Precision/train', pr if not np.isnan(pr) else 0, epoch)
        writer.add_scalar('Recall/train', re if not np.isnan(re) else 0, epoch)
        writer.add_scalar('F1/train', f1 if not np.isnan(f1) else 0, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Validation phase
        if val_loader is not None:
            # Set model to evaluation mode
            model.eval()

            # Clear previous progress bars and create a new one for validation
            tqdm.tqdm._instances.clear()
            pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
            pbar.set_description_str(f"VAL Epoch {epoch}")

            # Initialize validation metrics
            n = 0
            tn = 0
            fp = 0
            fn = 0
            tp = 0
            running_loss = 0.
            
            # Disable gradient computation for validation
            with torch.no_grad():
                for step, batch in pbar:
                    # Move batch data to device
                    x = batch["image"].to(device)
                    meta = batch["metadata"].to(device)
                    y = batch["label"].to(device)

                    # Forward pass based on selected mode
                    if fusion:
                        y_hat = model.forward(x, meta)
                    elif meta_only:
                        y_hat = model.forward(meta)
                    else:
                        y_hat = model.forward(x)
                    
                    # Binary predictions
                    y_pred = torch.round(torch.sigmoid(y_hat))

                    # Collect predictions and ground truth
                    if step == 0:
                        y_prob = torch.sigmoid(y_hat).cpu().detach()
                        y_true = y.cpu().detach()
                    else:
                        y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                        y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

                    # Calculate loss with focal loss
                    loss = loss_fxn(y_hat, y.view(-1, 1))

                    # Update validation metrics
                    running_loss += loss.item() * y.shape[0]
                    n += y.shape[0]
                    
                    # Update confusion matrix elements
                    for i in range(y.shape[0]):
                        if y[i] == 1 and y_pred[i] == 1:
                            tp += 1
                        if y[i] == 1 and y_pred[i] == 0:
                            fn += 1
                        if y[i] == 0 and y_pred[i] == 1:
                            fp += 1
                        if y[i] == 0 and y_pred[i] == 0:
                            tn += 1

                    # Calculate current metrics
                    l = running_loss / n
                    a = (tp + tn) / n
                    if tp + fp > 0:
                        pr = tp / (tp + fp)
                    else:
                        pr = np.nan
                    if tp + fn > 0:
                        re = tp / (tp + fn)
                    else:
                        re = np.nan
                    if tp + fp + fn > 0:
                        f1 = 2 * tp / (2 * tp + fp + fn)
                    else:
                        f1 = np.nan

                    # Update progress bar with current validation metrics
                    pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "f1": f1})
                    pbar.refresh()
                    time.sleep(0.01)

            # Calculate AUC metrics on entire validation set
            auc_roc = roc_auc_score(y_true, y_prob)
            prs, res, thrs = precision_recall_curve(y_true, y_prob)
            auc_pr = auc(res, prs)
            print("\tVal AUC ROC:", round(auc_roc, 3), "| Val AUC PR:", round(auc_pr, 3))

            # Store validation metrics in history
            history["val_loss"].append(l)
            history["val_auc_roc"].append(auc_roc)
            history["val_auc_pr"].append(auc_pr)
            history["val_acc"].append(a)
            history["val_precision"].append(pr)
            history["val_recall"].append(re)
            history["val_f1"].append(f1)
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/val', l, epoch)
            writer.add_scalar('Accuracy/val', a, epoch)
            writer.add_scalar('Precision/val', pr if not np.isnan(pr) else 0, epoch)
            writer.add_scalar('Recall/val', re if not np.isnan(re) else 0, epoch)
            writer.add_scalar('F1/val', f1 if not np.isnan(f1) else 0, epoch)
            
            # Save best model based on validation accuracy and positive F1 score
            if (a > best_val_acc) and (f1 > 0):
                best_val_acc = a
                best_model_state = model.state_dict().copy()  # Use .copy() instead of deepcopy
                print(f"\tNew best model saved with validation accuracy: {a:.4f} and F1: {f1:.4f}")

        # Increment epoch counter
        history["epoch"].append(epoch)
        
        # Apply learning rate scheduler step if provided
        if lr_scheduler is not None:
            lr_scheduler.step()
    
    # Close TensorBoard writer
    writer.close()
    
    # Load best model if a better one was found during training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation accuracy: {best_val_acc:.4f}")
        
    # Return trained model and training history
    return model, pd.DataFrame(history)

"""
# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# Create cosine learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epochs,  # Total number of epochs
    eta_min=min_lr     # Minimum learning rate at the end
)

# Train the model with the scheduler
model, history = train(
    model=model,
    train_loader=train_loader,
    max_epochs=max_epochs,
    optim=optimizer,
    device=device,
    val_loader=val_loader,
    fusion=True,
    meta_only=False,
    lr_scheduler=lr_scheduler
)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance problems
    
    Args:
        alpha: Class weight, can be a scalar (for positive samples) or tensor (weight for each class)
        gamma: Modulation factor, increases focus on hard-to-classify examples
        reduction: 'none' | 'mean' | 'sum'
        label_smoothing: Label smoothing parameter, 0 means no label smoothing
        dynamic_weight: Whether to calculate class weights dynamically based on batch distribution
    """
    def __init__(self, alpha=3.0, gamma=4.0, reduction='mean', label_smoothing=0, dynamic_weight=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.dynamic_weight = dynamic_weight
        
    def forward(self, inputs, targets):
        """
        Calculate Focal Loss
        
        Args:
            inputs: Model predictions (logits), shape [batch_size, 1]
            targets: Ground truth labels, shape [batch_size, 1]
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth_targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smooth_targets = targets
            
        # Calculate dynamic class weights
        if self.dynamic_weight:
            # Calculate class distribution in current batch
            pos_weight = (1 - targets.mean()) / (targets.mean() + 1e-6)
            # Limit weights to a reasonable range
            pos_weight = torch.clamp(pos_weight, min=1.0, max=5.0)
        else:
            pos_weight = torch.tensor(self.alpha, device=inputs.device)
        
        # Calculate initial loss using weighted BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, 
            smooth_targets, 
            reduction='none',
            pos_weight=pos_weight
        )
        
        # Get prediction probabilities
        probs = torch.sigmoid(inputs)
        # Adjust probabilities based on targets (use p for target=1, 1-p for target=0)
        pt = torch.where(smooth_targets > 0.5, probs, 1 - probs)
        
        # Calculate Focal Loss modulation factor
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight to BCE loss
        loss = focal_weight * bce_loss
        
        # Return result based on reduction mode
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # 'sum'
            return loss.sum()

def evaluate(model, test_loader, device, fusion=True, meta_only=False, focal_gamma=4, label_smoothing=0, dynamic_weight=True):
    """
    Evaluate a trained model on test data and print final performance metrics.
    
    Args:
        model: Trained neural network model to evaluate
        test_loader: DataLoader containing test data
        device: Device to run computations on (CPU/GPU)
        fusion: If True, use multimodal fusion with images and metadata
        meta_only: If True, use only metadata for prediction
        focal_gamma: Gamma parameter for Focal Loss (higher values increase focus on hard examples)
        label_smoothing: Label smoothing parameter
        dynamic_weight: Whether to calculate class weights dynamically
        
    Returns:
        results: Dictionary containing test metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Use the same Focal Loss as in training for consistency
    loss_fxn = FocalLoss(
        alpha=3.0,
        gamma=focal_gamma,
        reduction='mean',
        label_smoothing=label_smoothing,
        dynamic_weight=dynamic_weight
    ).to(device)
    
    # Create progress bar for test evaluation
    pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
    pbar.set_description_str("Testing")
    
    # Initialize test metrics
    n = 0           # Total samples processed
    tn = 0          # True negatives
    fp = 0          # False positives
    fn = 0          # False negatives 
    tp = 0          # True positives
    running_loss = 0.  # Cumulative loss
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for step, batch in pbar:
            # Move batch data to device
            x = batch["image"].to(device)
            meta = batch["metadata"].to(device)
            y = batch["label"].to(device)
            
            # Forward pass based on selected mode
            if fusion:
                y_hat = model.forward(x, meta)  # Multimodal fusion
            elif meta_only:
                y_hat = model.forward(meta)     # Metadata only
            else:
                y_hat = model.forward(x)        # Image only
            
            # Binary predictions
            y_pred = torch.round(torch.sigmoid(y_hat))
            
            # Collect predictions and ground truth
            if step == 0:
                y_prob = torch.sigmoid(y_hat).cpu().detach()
                y_true = y.cpu().detach()
            else:
                y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                y_true = torch.cat([y_true, y.cpu().detach()], dim=0)
            
            # Calculate loss with the same Focal Loss used in training
            loss = loss_fxn(y_hat, y.view(-1, 1))
            
            # Update test metrics
            running_loss += loss.item() * y.shape[0]
            n += y.shape[0]
            
            # Update confusion matrix elements
            for i in range(y.shape[0]):
                if y[i] == 1 and y_pred[i] == 1:
                    tp += 1
                if y[i] == 1 and y_pred[i] == 0:
                    fn += 1
                if y[i] == 0 and y_pred[i] == 1:
                    fp += 1
                if y[i] == 0 and y_pred[i] == 0:
                    tn += 1
    
    # Calculate final metrics
    loss = running_loss / n
    accuracy = (tp + tn) / n
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    
    # Recall/Sensitivity
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    # F1 score
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    
    # Calculate AUC metrics
    auc_roc = roc_auc_score(y_true, y_prob)
    prs, res, thrs = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(res, prs)
    
    # Print comprehensive evaluation results
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY:")
    print("="*80)
    print(f"Loss:       {loss:.4f}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUC-ROC:    {auc_roc:.4f}")
    print(f"AUC-PR:     {auc_pr:.4f}")
    print("="*80)
    
    # Calculate and print confusion matrix
    cm = confusion_matrix(y_true, y_prob > 0.5)
    print("\nConfusion Matrix:")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")
    print("="*80)
    
    # Calculate class-specific metrics for imbalanced classification
    print("\nClass Imbalance Metrics:")
    print(f"Class 0 (Majority) Accuracy: {tn/(tn+fp):.4f}")
    print(f"Class 1 (Minority) Accuracy: {tp/(tp+fn):.4f}")
    print(f"Balanced Accuracy: {(recall + specificity)/2:.4f}")
    
    # Calculate optimal threshold (if needed)
    optimal_idx = np.argmax(np.array(res) + np.array(prs))
    optimal_threshold = thrs[optimal_idx] if len(thrs) > optimal_idx else 0.5
    print(f"Optimal Decision Threshold: {optimal_threshold:.4f}")
    print("="*80)
    
    # Store results in dictionary
    results = {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "balanced_accuracy": (recall + specificity)/2,
        "optimal_threshold": optimal_threshold
    }
    
    return results

def train_with_calibration(model, train_loader, max_epochs, optim, device, val_loader, 
                    calibration_method='isotonic', fusion=True, meta_only=False, 
                    lr_scheduler=None, label_smoothing=0.0, focal_gamma=10.0, dynamic_weight=True):
    """
    Train a model with Focal Loss, probability calibration, enhanced metrics and visualization.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader containing training data
        max_epochs: Maximum number of epochs to train for
        optim: Optimizer for parameter updates
        device: Device to run computations on (CPU/GPU)
        val_loader: DataLoader containing validation data (optional)
        calibration_method: Method for probability calibration ('isotonic', 'sigmoid', or 'platt')
        fusion: If True, use multimodal fusion with images and metadata
        meta_only: If True, use only metadata for prediction
        lr_scheduler: Cosine decay learning rate scheduler
        label_smoothing: Label smoothing parameter
        focal_gamma: Gamma parameter for Focal Loss (higher values increase focus on hard examples)
        dynamic_weight: Whether to calculate class weights dynamically
        
    Returns:
        model: Trained model with best weights according to validation accuracy
        calibrated_model: Model wrapped with probability calibration
        history: DataFrame containing training metrics history
    """
    import numpy as np
    import pandas as pd
    import torch
    import tqdm
    import time
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
    from sklearn.metrics import log_loss, roc_curve, precision_recall_fscore_support
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import io
    from torchvision.transforms import ToTensor
    
    # Dictionary to store training and validation metrics for each epoch
    history = {
        "epoch": [], "loss": [], "auc_roc": [], "auc_pr": [], "acc": [], 
        "precision": [], "recall": [], "f1": [], "brier_score": [], "log_loss": [],
        "val_loss": [], "val_auc_roc": [], "val_auc_pr": [], "val_acc": [], 
        "val_precision": [], "val_recall": [], "val_f1": [], 
        "val_brier_score": [], "val_log_loss": [], "lr": []
    }
    
    # Initialize Focal Loss
    loss_fxn = FocalLoss(
        alpha=3.0,
        gamma=focal_gamma,
        reduction='mean',
        label_smoothing=label_smoothing,
        dynamic_weight=dynamic_weight
    ).to(device)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter()
    
    # Variables to track best model
    best_val_auc = 0.0
    best_model_state = None
    
    # Store calibration data
    calibration_X = []
    calibration_y = []
    
    # Main training loop
    for epoch in range(1, max_epochs + 1):
        # Set model to training mode
        model.train()

        # Create progress bar for current epoch
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        pbar.set_description_str(f"Epoch {epoch}")

        # Initialize metrics for current epoch
        n = 0  # Total samples processed
        tn = 0  # True negatives
        fp = 0  # False positives
        fn = 0  # False negatives
        tp = 0  # True positives
        running_loss = 0.  # Cumulative loss
        
        # Iterate through batches
        for step, batch in pbar:
            # Zero gradients before forward pass
            optim.zero_grad()

            # Move batch data to device
            x = batch["image"].to(device)
            meta = batch["metadata"].to(device)
            y = batch["label"].to(device)

            # Forward pass based on selected mode (fusion/meta_only/image_only)
            if fusion:
                y_hat = model.forward(x, meta)  # Multimodal fusion
            elif meta_only:
                y_hat = model.forward(meta)  # Metadata only
            else:
                y_hat = model.forward(x)  # Image only
            
            # Binary predictions
            y_prob_raw = torch.sigmoid(y_hat)
            y_pred = torch.round(y_prob_raw)

            # Collect predictions and ground truth for AUC calculation
            if step == 0:
                y_prob = y_prob_raw.cpu().detach()
                y_true = y.cpu().detach()
            else:
                y_prob = torch.cat([y_prob, y_prob_raw.cpu().detach()], dim=0)
                y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

            # Calculate loss with focal loss
            loss = loss_fxn(y_hat, y.view(-1, 1))
                
            # Backward pass and optimization
            loss.backward()
            optim.step()  # Update model parameters
            
            # Update running metrics
            running_loss += loss.item() * y.shape[0]
            n += y.shape[0]
            
            # Calculate confusion matrix elements
            for i in range(y.shape[0]):
                if y[i] == 1 and y_pred[i] == 1:
                    tp += 1
                if y[i] == 1 and y_pred[i] == 0:
                    fn += 1
                if y[i] == 0 and y_pred[i] == 1:
                    fp += 1
                if y[i] == 0 and y_pred[i] == 0:
                    tn += 1

            # Calculate current metrics
            l = running_loss / n  # Average loss
            a = (tp + tn) / n  # Accuracy
            
            # Precision (positive predictive value)
            if tp + fp > 0:
                pr = tp / (tp + fp)
            else:
                pr = np.nan
                
            # Recall/Sensitivity (true positive rate)    
            if tp + fn > 0:
                re = tp / (tp + fn)
            else:
                re = np.nan
                
            # F1 score
            if tp + fp + fn > 0:
                f1 = 2 * tp / (2 * tp + fp + fn)
            else:
                f1 = np.nan

            # Update progress bar with current metrics
            pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "f1": f1})
            pbar.refresh()
            time.sleep(0.01)

        # Convert to numpy arrays for sklearn metrics
        y_true_np = y_true.numpy()
        y_prob_np = y_prob.numpy()
        
        # Calculate AUC metrics on entire training set
        auc_roc = roc_auc_score(y_true_np, y_prob_np)  # Area under ROC curve
        prs, res, thrs = precision_recall_curve(y_true_np, y_prob_np)
        auc_pr = auc(res, prs)  # Area under Precision-Recall curve
        
        # Additional calibration metrics
        brier = brier_score_loss(y_true_np, y_prob_np)  # Brier score
        logloss = log_loss(y_true_np, y_prob_np)  # Log loss
        
        print(f"\tAUC ROC: {auc_roc:.3f} | AUC PR: {auc_pr:.3f} | Brier: {brier:.3f} | LogLoss: {logloss:.3f}")

        # Store training metrics in history
        history["loss"].append(l)
        history["auc_roc"].append(auc_roc)
        history["auc_pr"].append(auc_pr)
        history["acc"].append(a)
        history["precision"].append(pr)
        history["recall"].append(re)
        history["f1"].append(f1)
        history["brier_score"].append(brier)
        history["log_loss"].append(logloss)
        
        # Record current learning rate
        current_lr = optim.param_groups[0]['lr']
        history["lr"].append(current_lr)

        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', l, epoch)
        writer.add_scalar('Accuracy/train', a, epoch)
        writer.add_scalar('AUC/ROC/train', auc_roc, epoch)
        writer.add_scalar('AUC/PR/train', auc_pr, epoch)
        writer.add_scalar('Precision/train', pr if not np.isnan(pr) else 0, epoch)
        writer.add_scalar('Recall/train', re if not np.isnan(re) else 0, epoch)
        writer.add_scalar('F1/train', f1 if not np.isnan(f1) else 0, epoch)
        writer.add_scalar('Brier/train', brier, epoch)
        writer.add_scalar('LogLoss/train', logloss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Validation phase
        if val_loader is not None:
            # Set model to evaluation mode
            model.eval()

            # Clear previous progress bars and create a new one for validation
            tqdm.tqdm._instances.clear()
            pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
            pbar.set_description_str(f"VAL Epoch {epoch}")

            # Initialize validation metrics
            n = 0
            tn = 0
            fp = 0
            fn = 0
            tp = 0
            running_loss = 0.
            
            # Disable gradient computation for validation
            with torch.no_grad():
                for step, batch in pbar:
                    # Move batch data to device
                    x = batch["image"].to(device)
                    meta = batch["metadata"].to(device)
                    y = batch["label"].to(device)

                    # Forward pass based on selected mode
                    if fusion:
                        y_hat = model.forward(x, meta)
                    elif meta_only:
                        y_hat = model.forward(meta)
                    else:
                        y_hat = model.forward(x)
                    
                    # Binary predictions
                    y_prob_raw = torch.sigmoid(y_hat)
                    y_pred = torch.round(y_prob_raw)

                    # Collect predictions and ground truth
                    if step == 0:
                        y_prob = y_prob_raw.cpu().detach()
                        y_true = y.cpu().detach()
                    else:
                        y_prob = torch.cat([y_prob, y_prob_raw.cpu().detach()], dim=0)
                        y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

                    # Calculate loss with focal loss
                    loss = loss_fxn(y_hat, y.view(-1, 1))

                    # Update validation metrics
                    running_loss += loss.item() * y.shape[0]
                    n += y.shape[0]
                    
                    # Update confusion matrix elements
                    for i in range(y.shape[0]):
                        if y[i] == 1 and y_pred[i] == 1:
                            tp += 1
                        if y[i] == 1 and y_pred[i] == 0:
                            fn += 1
                        if y[i] == 0 and y_pred[i] == 1:
                            fp += 1
                        if y[i] == 0 and y_pred[i] == 0:
                            tn += 1

                    # Calculate current metrics
                    l = running_loss / n
                    a = (tp + tn) / n
                    if tp + fp > 0:
                        pr = tp / (tp + fp)
                    else:
                        pr = np.nan
                    if tp + fn > 0:
                        re = tp / (tp + fn)
                    else:
                        re = np.nan
                    if tp + fp + fn > 0:
                        f1 = 2 * tp / (2 * tp + fp + fn)
                    else:
                        f1 = np.nan

                    # Update progress bar with current validation metrics
                    pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "f1": f1})
                    pbar.refresh()
                    time.sleep(0.01)

            # Convert to numpy arrays for sklearn metrics
            y_true_np = y_true.numpy()
            y_prob_np = y_prob.numpy()
            
            # Keep validation data for calibration in the last epoch
            if epoch == max_epochs:
                calibration_X = y_prob_np
                calibration_y = y_true_np
                
            # Calculate AUC metrics on entire validation set
            auc_roc = roc_auc_score(y_true_np, y_prob_np)
            prs, res, thrs = precision_recall_curve(y_true_np, y_prob_np)
            auc_pr = auc(res, prs)
            
            # Additional calibration metrics
            brier = brier_score_loss(y_true_np, y_prob_np)
            logloss = log_loss(y_true_np, y_prob_np)
            
            print(f"\tVal AUC ROC: {auc_roc:.3f} | Val AUC PR: {auc_pr:.3f} | Val Brier: {brier:.3f} | Val LogLoss: {logloss:.3f}")

            # Store validation metrics in history
            history["val_loss"].append(l)
            history["val_auc_roc"].append(auc_roc)
            history["val_auc_pr"].append(auc_pr)
            history["val_acc"].append(a)
            history["val_precision"].append(pr)
            history["val_recall"].append(re)
            history["val_f1"].append(f1)
            history["val_brier_score"].append(brier)
            history["val_log_loss"].append(logloss)
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/val', l, epoch)
            writer.add_scalar('Accuracy/val', a, epoch)
            writer.add_scalar('AUC/ROC/val', auc_roc, epoch)
            writer.add_scalar('AUC/PR/val', auc_pr, epoch)
            writer.add_scalar('Precision/val', pr if not np.isnan(pr) else 0, epoch)
            writer.add_scalar('Recall/val', re if not np.isnan(re) else 0, epoch)
            writer.add_scalar('F1/val', f1 if not np.isnan(f1) else 0, epoch)
            writer.add_scalar('Brier/val', brier, epoch)
            writer.add_scalar('LogLoss/val', logloss, epoch)
            
            # Create and log calibration curve plot (in last epoch)
            if epoch == max_epochs:
                fig = plot_calibration_curve(y_true_np, y_prob_np)
                writer.add_figure('Calibration/Curve', fig, epoch)
                
                # Create and log ROC curve
                fig_roc = plot_roc_curve(y_true_np, y_prob_np)
                writer.add_figure('ROC/Curve', fig_roc, epoch)
                
                # Create and log Precision-Recall curve
                fig_pr = plot_pr_curve(y_true_np, y_prob_np)
                writer.add_figure('PR/Curve', fig_pr, epoch)
            
            # Save best model based on validation AUC-ROC score
            if auc_roc > best_val_auc:
                best_val_auc = auc_roc
                best_model_state = model.state_dict().copy()  # Use .copy() instead of deepcopy
                print(f"\tNew best model saved with validation AUC-ROC: {auc_roc:.4f} and F1: {f1:.4f}")

        # Increment epoch counter
        history["epoch"].append(epoch)
        
        # Apply learning rate scheduler step if provided
        if lr_scheduler is not None:
            lr_scheduler.step()
    
    # Close TensorBoard writer
    writer.close()
    
    # Load best model if a better one was found during training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation AUC-ROC: {best_val_auc:.4f}")
        model_save_path = f"5year_best_model_auc{best_val_auc:.4f}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved to {model_save_path}")
    
    # Apply probability calibration (if we have validation data)
    calibrated_model = None
    if len(calibration_X) > 0:
        print(f"Applying {calibration_method} calibration to model predictions...")
        calibrated_model = create_calibrated_model(model, calibration_method, calibration_X, 
                                                 calibration_y, fusion, meta_only)
        
        # Generate calibrated predictions
        calibrated_probs = None
        try:
            # Try to get calibrated probabilities
            calibrated_probs = calibrated_model.predict_proba(calibration_X)[:, 1]
            print("Successfully generated calibrated probabilities")
        except Exception as e:
            print(f"Warning: Could not generate calibrated probabilities: {str(e)}")
            print("Using raw probabilities for visualization")
            calibrated_probs = calibration_X  # Fallback to uncalibrated
            
        # Plot final calibration curves after calibration
        fig = plot_calibration_comparison(calibration_y, calibration_X, calibrated_probs)
        fig.savefig('5year_calibration_comparison.png')
        print("Calibration comparison plot saved to 'calibration_comparison.png'")
        
        # Save the calibrated model
        calibrated_model_path = f"5year_calibrated_model_{calibration_method}.pkl"
        import pickle
        with open(calibrated_model_path, 'wb') as f:
            pickle.dump(calibrated_model, f)
        print(f"Calibrated model saved to {calibrated_model_path}")
    else:
        calibrated_model = model
        print("Warning: No validation data available for calibration")
    
    # Return trained model, calibrated model, and training history
    return model, calibrated_model, pd.DataFrame(history)

class CalibratedModelWrapper:
    """
    A wrapper class for calibrated models that can be saved with pickle.
    """
    def __init__(self, base_model, calibrator, fusion=True, meta_only=False):
        self.base_model = base_model
        self.calibrator = calibrator
        self.fusion = fusion
        self.meta_only = meta_only
        
    def predict_proba(self, X):
        """Return calibrated probabilities"""
        import numpy as np
        
        # For prediction from raw probabilities (used during validation)
        if isinstance(X, np.ndarray) and X.ndim == 1:
            probs = X.reshape(-1, 1)
            calibrated_probs = self.calibrator.predict(probs).reshape(-1)
            # Convert to sklearn expected format (prob_class_0, prob_class_1)
            return np.vstack([1-calibrated_probs, calibrated_probs]).T
        
        # For prediction from raw probabilities (non-reshaped array)
        elif isinstance(X, np.ndarray) and X.shape[1] == 1:
            calibrated_probs = self.calibrator.predict(X).reshape(-1)
            return np.vstack([1-calibrated_probs, calibrated_probs]).T
        
        # X is already predicted probabilities from the original model
        # This handles the case when we're plotting calibration comparisons
        elif len(X.shape) == 1:
            probs = X.reshape(-1, 1)
            calibrated_probs = self.calibrator.predict(probs).reshape(-1)
            return np.vstack([1-calibrated_probs, calibrated_probs]).T
        
        # For future implementation: direct model prediction
        else:
            # This part would handle feature inputs requiring model forward pass
            # For now we raise a more helpful error
            raise ValueError("This calibrated model wrapper currently only supports probability inputs, " 
                           + "not raw features. Use the base model to get probabilities first.")
        
    def predict(self, X, threshold=0.5):
        """Return binary predictions"""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

def create_calibrated_model(model, calibration_method, X_val, y_val, fusion=True, meta_only=False):
    """
    Create a calibrated model wrapper around the trained model.
    
    Args:
        model: Trained PyTorch model
        calibration_method: Calibration method ('isotonic', 'sigmoid', or 'platt')
        X_val: Validation set predictions
        y_val: Validation set true labels
        fusion: Whether to use multimodal fusion
        meta_only: Whether to use only metadata
        
    Returns:
        A sklearn-compatible calibrated model
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    
    # Train the calibrator based on the chosen method
    if calibration_method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(X_val.reshape(-1, 1), y_val)
    elif calibration_method in ['sigmoid', 'platt']:
        # Platt scaling is logistic regression on the scores
        calibrator = LogisticRegression(C=1.0, solver='lbfgs')
        calibrator.fit(X_val.reshape(-1, 1), y_val)
    else:
        raise ValueError(f"Unknown calibration method: {calibration_method}")
        
    # Return the calibrated model
    return CalibratedModelWrapper(model, calibrator, fusion, meta_only)


def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """
    Plot calibration curve for the model.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for the calibration curve
        
    Returns:
        Matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    
    # Create calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot perfectly calibrated
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    
    # Plot model calibration
    ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    
    # Formatting
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig


def plot_roc_curve(y_true, y_prob):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    
    # Calculate ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
    
    # Plot random guess line
    ax.plot([0, 1], [0, 1], linestyle='--', label='Random guess')
    
    # Formatting
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    return fig


def plot_pr_curve(y_true, y_prob):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot PR curve
    ax.plot(recall, precision, label=f'PR curve (AP = {ap_score:.3f})')
    
    # Plot random baseline
    baseline = sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    # Formatting
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig


def plot_calibration_comparison(y_true, y_prob_raw, y_prob_cal=None, n_bins=10):
    """
    Plot calibration curve for both the original and calibrated model.
    
    Args:
        y_true: True labels
        y_prob_raw: Original predicted probabilities
        y_prob_cal: Calibrated predicted probabilities (if None, will use calibrated_model to predict)
        n_bins: Number of bins for the calibration curve
        
    Returns:
        Matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    import numpy as np
    
    # Create calibration curves
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_prob_raw, n_bins=n_bins)
    
    # Handle calibrated probabilities
    if y_prob_cal is not None:
        # If calibrated probabilities are directly provided
        if isinstance(y_prob_cal, np.ndarray) and y_prob_cal.ndim == 2:
            # If 2D array with shape (n_samples, 2), extract positive class probs
            cal_probs = y_prob_cal[:, 1]
        else:
            # Use as is (assuming 1D array of positive class probabilities)
            cal_probs = y_prob_cal
    else:
        # If no calibrated probabilities provided, just use raw (for testing)
        cal_probs = y_prob_raw
    
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, cal_probs, n_bins=n_bins)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot perfectly calibrated
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
    
    # Plot original model calibration
    ax.plot(prob_pred_raw, prob_true_raw, marker='o', linewidth=1, color='red', 
            label='Original model')
    
    # Plot calibrated model
    ax.plot(prob_pred_cal, prob_true_cal, marker='s', linewidth=1, color='blue', 
            label='Calibrated model')
    
    # Formatting
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve Comparison')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig

import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

from dataset import HipFusionDataset

def parse_args():
    """
    Parse command line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a Random Forest model on HipFusion dataset using metadata only')
    
    # Data paths
    parser.add_argument('--train_img_path', type=str, required=True, help='Directory path containing training .nii or .nii.gz files')
    parser.add_argument('--train_label_path', type=str, required=True, help='Path to Excel file containing training labels and metadata')
    parser.add_argument('--validation_img_path', type=str, required=True, help='Directory path containing validation .nii or .nii.gz files')
    parser.add_argument('--validation_label_path', type=str, required=True, help='Path to Excel file containing validation labels and metadata')
    parser.add_argument('--test_img_path', type=str, required=True, help='Directory path containing test .nii or .nii.gz files')
    parser.add_argument('--test_label_path', type=str, required=True, help='Path to Excel file containing test labels and metadata')
    
    # Training parameters
    parser.add_argument('--augment', action='store_true', help='Whether to apply data augmentation (only for training set)')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save model and results')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--tune', action='store_true', help='Whether to perform hyperparameter tuning')
    
    # Model hyperparameters (used if tune=False)
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the random forest')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the trees')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum samples required to split an internal node')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='Minimum samples required to be at a leaf node')
    
    return parser.parse_args()



