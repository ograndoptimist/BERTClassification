import torch


def categorical_accuracy(preds, y):
    """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def fit(model,
        data,
        optimizer,
        criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for step in range(100):
        for batch_x, batch_y in data:
            predictions = model(batch_x)

            loss = criterion(predictions, batch_y)
            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()

            acc = categorical_accuracy(predictions, batch_y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / 100, epoch_acc / 100


def evaluate(model,
             data,
             criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for step in range(100):
            for batch_x, batch_y in data:
                predictions = model(batch_x)

                loss = criterion(predictions, batch_y)
                acc = categorical_accuracy(predictions, batch_y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
    return epoch_loss / 100, epoch_acc / 100
