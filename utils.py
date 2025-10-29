import torch

def train(model, dataloader, dataset_size, optimizer, loss_fn, device):
    """
    Huấn luyện mô hình cho một epoch.
    Trả về Loss và Accuracy của tập Train.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
    # Trả về Loss và Accuracy của tập Train
    return epoch_loss, epoch_acc 


def validate(model, dataloader, dataset_size, loss_fn, device):
    """
    Đánh giá mô hình trên tập validation/test.
    Trả về Loss và Accuracy của tập Validation.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
    # TRẢ VỀ val_loss và val_acc
    return epoch_loss, epoch_acc