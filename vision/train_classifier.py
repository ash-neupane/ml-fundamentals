import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

from mlp import MLP
from mnist_data import load_data 
from utils import plot_classes_preds, images_to_probs

train_loader, test_loader = load_data()

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
writer = SummaryWriter()

# Training loop
epochs = 5
running_loss = 0.0
for epoch in range(epochs):

    epoch_start_time = time.time()

    for i, data  in enumerate(train_loader, 0):
        images, labels = data
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:    # every 1000 mini-batches...
            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(train_loader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(model, images, labels),
                            global_step=epoch * len(train_loader) + i)
            running_loss = 0.0


    epoch_duration = (time.time() - epoch_start_time)   
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Time: {epoch_duration} sec')
