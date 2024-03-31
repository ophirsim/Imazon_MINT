import torch

def train(train_loader, val_loader, model, loss_fn, optimizer, acc_metrics, epochs):
    size = len(train_loader.dataset)
    model.train()
    train_losses, val_losses = [], []
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
            if batch % 200 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        _, train_loss = test(train_loader, model, loss_fn, acc_metrics)
        val_accs, val_loss = test(val_loader, model, loss_fn, acc_metrics)
        print(f"Epoch {epoch} Train loss: {train_loss}")
        print(f"Epoch {epoch} Val loss: {val_loss}")
        for metric in val_accs:
            print(f"Epoch {epoch} Val {metric}: {val_accs[metric]}")

        model.train()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_losses,
            'val_loss_history': val_losses,
            }, f'model_{epoch}.ckpt')
        
    return train_losses, val_losses



def test(dataloader, model, loss_fn, acc_metrics):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    
    test_accs = {}

    for metric in acc_metrics:
        test_accs[metric] = 0

    with torch.no_grad():
        for X, y, gap in dataloader:
            # Compute prediction loss and accuracy
            pred = model(X, gap)
            test_loss += loss_fn(pred, y).item()
            for metric in acc_metrics:
                test_accs[metric] += acc_metrics[metric](pred, y)

    # normalize loss and accuracy
    test_loss /= num_batches
    for metric in acc_metrics:
        test_accs[metric] /= size
    return test_accs, test_loss