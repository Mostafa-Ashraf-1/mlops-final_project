import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return (preds == labels).float().mean()

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels)
        return loss, acc

    def training_epoch_end(self, outputs):
        batch_accs = outputs
        epoch_acc = torch.stack(batch_accs).mean()
        return epoch_acc.item()
    
    def validation_step(self, batch):
        data, labels = batch
        out = self(data)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss' : loss, 'val_acc' : acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
            .format(
                epoch,
                result['train_acc'],
                result['val_loss'],
                result['val_acc']
            )
        )
