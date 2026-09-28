import torch
from torch import nn
from src.final_project.DL_Classes import ImageClassificationBase
import mlflow
import torch.nn.functional as F
from torchvision import models

@torch.no_grad()
def evaluate(model, val_loader, n_bins=15, threshold=0.8):
    model.eval()

    all_probs = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    for images, labels in val_loader:
        outputs = model(images)

        loss = F.cross_entropy(outputs, labels)
        probs = F.softmax(outputs, dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_probs.append(probs)
        all_labels.append(labels)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    # --------------------------------------------------
    # Predictions + confidence
    # --------------------------------------------------
    confidences, predictions = probs.max(dim=1)

    correct = predictions.eq(labels)

    # --------------------------------------------------
    # Top-1 Accuracy
    # --------------------------------------------------
    top1 = correct.float().mean().item()

    # --------------------------------------------------
    # Macro-F1
    # --------------------------------------------------
    num_classes = probs.size(1)

    f1_scores = []

    for class_idx in range(num_classes):
        tp = ((predictions == class_idx) & (labels == class_idx)).sum()
        fp = ((predictions == class_idx) & (labels != class_idx)).sum()
        fn = ((predictions != class_idx) & (labels == class_idx)).sum()

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else torch.tensor(0.0, device=probs.device)
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else torch.tensor(0.0, device=probs.device)
        )

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = torch.tensor(0.0, device=probs.device)

        f1_scores.append(f1)

    macro_f1 = torch.stack(f1_scores).mean().item()

    # --------------------------------------------------
    # Coverage
    # --------------------------------------------------
    selected = confidences >= threshold

    coverage = selected.float().mean().item()

    # --------------------------------------------------
    # Selective Accuracy
    # --------------------------------------------------
    if selected.any():
        selective_accuracy = correct[selected].float().mean().item()
    else:
        selective_accuracy = 0.0

    # --------------------------------------------------
    # ECE
    # --------------------------------------------------
    accuracies = correct

    bin_boundaries = torch.linspace(
        0, 1, n_bins + 1, device=probs.device
    )

    ece = torch.zeros(1, device=probs.device)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        in_bin = (
            (confidences > lower)
            & (confidences <= upper)
        )

        if in_bin.any():
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = accuracies[in_bin].float().mean()

            ece += (
                torch.abs(bin_confidence - bin_accuracy)
                * in_bin.float().mean()
            )

    return {
        "val_loss": total_loss / total_samples,
        "val_acc": top1,
        "top1": top1,
        "macro_f1": macro_f1,
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "ece": ece.item(),
    }


def fit(
    epochs,
    lr,
    model,
    train_loader,
    val_loader,
    opt_func=torch.optim.SGD,
    weight_decay=0
):
    optimizer = opt_func(
        model.parameters(),
        lr,
        weight_decay=weight_decay
    )

    history = []

    for epoch in range(epochs):

        # --------------------------------------------------
        # Training
        # --------------------------------------------------
        model.train()

        train_losses = []
        train_accs = []

        for batch in train_loader:
            loss, acc = model.training_step(batch)

            train_losses.append(loss)
            train_accs.append(acc)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = torch.stack(train_losses).mean().item()
        train_acc = torch.stack(train_accs).mean().item()

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        result = evaluate(model, val_loader)

        val_loss = result["val_loss"]
        val_acc = result["val_acc"]
        top1 = result["top1"]
        macro_f1 = result["macro_f1"]
        coverage = result["coverage"]
        selective_accuracy = result["selective_accuracy"]
        ece = result["ece"]

        # --------------------------------------------------
        # History
        # --------------------------------------------------
        result["train_loss"] = train_loss
        result["train_acc"] = train_acc

        history.append(result)

        # --------------------------------------------------
        # MLflow
        # --------------------------------------------------
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,

                "val_loss": val_loss,
                "val_accuracy": val_acc,

                "top1": top1,
                "macro_f1": macro_f1,

                "coverage": coverage,
                "selective_accuracy": selective_accuracy,

                "ece": ece,

                "weight_decay": weight_decay,
            },
            step=epoch
        )

        model.epoch_end(epoch, result)

    return history


class PetsModel(ImageClassificationBase):
    def __init__(self, num_classes, model, pretrained=True):
        super().__init__()

        if model != models.mobilenet_v3_small:
            # Use pretrained model
            self.network = model(pretrained=pretrained)
            for param in self.network.parameters():
                param.requires_grad = False

            self.network.fc = nn.Linear(
                self.network.fc.in_features,
                num_classes
            )
            for param in self.network.fc.parameters():
                param.requires_grad_(True)

            for param in self.network.layer4[0].parameters():
                param.requires_grad_(True)

            # Replace last layer
            self.network.layer1[0].register_forward_hook(lambda m, input, out: nn.functional.dropout2d(out, p=0.2, training=m.training))
            self.network.layer2[0].register_forward_hook(lambda m, input, out: nn.functional.dropout2d(out, p=0.2, training=m.training))
            self.network.layer3[0].register_forward_hook(lambda m, input, out: nn.functional.dropout2d(out, p=0.2, training=m.training))
            self.network.layer4[0].register_forward_hook(lambda m, input, out: nn.functional.dropout2d(out, p=0.2, training=m.training))

            self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

        else:
            self.network = model(pretrained=pretrained)

            # 1. Freeze all parameters first
            for param in self.network.parameters():
                param.requires_grad = False

            in_features = self.network.classifier[3].in_features
            self.network.classifier[3] = nn.Linear(in_features, num_classes)

            self.network.classifier.requires_grad_(True)
            self.network.features[-1].requires_grad_(True)  # Unfreezes the final conv block

            self.network.features[3].register_forward_hook(
                        lambda m, inp, out: nn.functional.dropout2d(out, p=0.2, training=m.training)
                    )
            self.network.features[6].register_forward_hook(
                        lambda m, inp, out: nn.functional.dropout2d(out, p=0.2, training=m.training)
                    )
            self.network.features[9].register_forward_hook(
                        lambda m, inp, out: nn.functional.dropout2d(out, p=0.2, training=m.training)
                    )
            
    def forward(self, xb):
        return self.network(xb)