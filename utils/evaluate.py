import torch
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.utils.data.distributed
import torch.nn as nn

def compute_accuracy(model, test_set, get_confusion_matrix=False, device="cuda"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss = 0.0

    if type(test_set) == type([1]):
        pass
    else:
        test_set = [test_set]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in test_set:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)
                loss = criterion(out, target)
                test_loss += loss

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return round(100*correct/float(total), 4), conf_matrix

    return round(100*correct/float(total), 4), test_loss/(batch_idx+1)
