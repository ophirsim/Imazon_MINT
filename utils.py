import torch

def train(train_loader, val_loader, model, loss_fn, optimizer, acc_metric, epochs):
    train_losses, val_losses = [], []
    size = len(train_loader.dataset)
    model.train()
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for batch, (X, y, gap) in enumerate(train_loader):
            # Compute prediction error
            pred = model(X, gap)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # document loss
            if batch % 5 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        _, train_loss = test(train_loader, model, loss_fn, acc_metric)
        _, val_loss = test(val_loader, model, loss_fn, acc_metric)
        print(f"Epoch {epoch} Train loss: {train_loss}")
        print(f"Epoch {epoch} Val loss: {val_loss}")
        model.train()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return train_losses, val_losses



def test(dataloader, model, loss_fn, acc_metric):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.no_grad():
        for X, y, gap in dataloader:
            # Compute prediction loss and accuracy
            pred = model(X, gap)
            test_loss += loss_fn(pred, y).item()
            test_acc += acc_metric(pred, y)

    # normalize loss and accuracy
    test_loss /= num_batches
    test_acc /= size
    return test_acc, test_loss