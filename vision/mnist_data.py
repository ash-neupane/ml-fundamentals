from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data():
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t*2 - 1)),
    ])

    # Download and load the training data
    train_set = datasets.MNIST('./mnist/', download=True, train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Download and load the test data
    test_set = datasets.MNIST('./mnist/', download=True, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    return train_loader, test_loader