import numpy as np
import torch
import tqdm
import pandas as pd
import random
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix

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


def train(model, train_loader, max_epochs, optim, device, val_loader, fusion=True, meta_only=False, lr_scheduler=None, label_smoothing=0.1):
    """
    Train a model with cosine learning rate decay and TensorBoard logging.
    
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
        
    Returns:
        model: Trained model with best weights according to validation accuracy
        history: DataFrame containing training metrics history
    """
    # Dictionary to store training and validation metrics for each epoch
    history = {"epoch": [], "loss": [], "auc_roc": [], "auc_pr": [], "acc": [], "precision": [], "recall": [], "f1": [],
               "val_loss": [], "val_auc_roc": [], "val_auc_pr": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": [], "lr": []}
    
    # Define CrossEntropyLoss for binary classification
    loss_fxn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3).to(device))
    
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
            
            y_pred = torch.round(torch.sigmoid(y_hat))

            if step == 0:
                y_prob = torch.sigmoid(y_hat).cpu().detach()
                y_true = y.cpu().detach()
            else:
                y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

            # Calculate loss
            if label_smoothing != 0:
               loss = loss_fxn(y_hat, (y*(1-label_smoothing)+0.5*label_smoothing).view(-1,1))
            else:
                # Standard loss calculation without label smoothing
                loss = loss_fxn(y_hat, y.view(-1, 1))
                
            # Backward pass and optimization
            loss.mean().backward()
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
                
            # Specificity (true negative rate)
            if tn + fp > 0:
                sp = tn / (tn + fp)
            else:
                sp = np.nan
                
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
                    
                    # Transform logits for CrossEntropyLoss
                    
                    # Binary predictions
                    y_pred = torch.round(torch.sigmoid(y_hat))

                    # Collect predictions and ground truth
                    if step == 0:
                        y_prob = torch.sigmoid(y_hat).cpu().detach()
                        y_true = y.cpu().detach()
                    else:
                        y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                        y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

                    # Calculate loss
                    if label_smoothing != 0:
                        loss = loss_fxn(y_hat, (y*(1-label_smoothing)+0.5*label_smoothing).view(-1,1))
                    else:
                        # Standard loss calculation without label smoothing
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
            
            # Save best model based on validation accuracy
            if a > best_val_acc:
                best_val_acc = a
                best_model_state = model.state_dict().copy()  # Use .copy() instead of deepcopy
                print(f"\tNew best model saved with validation accuracy: {a:.4f}")

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

def evaluate(model, test_loader, device, fusion=True, meta_only=False):
    """
    Evaluate a trained model on test data and print final performance metrics.
    
    Args:
        model: Trained neural network model to evaluate
        test_loader: DataLoader containing test data
        device: Device to run computations on (CPU/GPU)
        fusion: If True, use multimodal fusion with images and metadata
        meta_only: If True, use only metadata for prediction
        
    Returns:
        results: Dictionary containing test metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Define CrossEntropyLoss for evaluation
    loss_fxn = torch.nn.CrossEntropyLoss()
    
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
            
            # Transform logits for CrossEntropyLoss
            y_hat_reshaped = torch.cat([-y_hat, y_hat], dim=1)
            y_labels = y.long()
            
            # Binary predictions
            y_pred = torch.round(torch.sigmoid(y_hat))
            
            # Collect predictions and ground truth
            if step == 0:
                y_prob = torch.sigmoid(y_hat).cpu().detach()
                y_true = y.cpu().detach()
            else:
                y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                y_true = torch.cat([y_true, y.cpu().detach()], dim=0)
            
            # Calculate loss
            loss = loss_fxn(y_hat_reshaped, y_labels)
            
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
        "true_positives": tp
    }
    
    return results