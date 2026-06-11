import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from data.loader import MnistDataloader
from os.path import join


class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(data_dir="data/raw", batch_size=64):
    train_img = join(data_dir, "train-images-idx3-ubyte/train-images-idx3-ubyte")
    train_lbl = join(data_dir, "train-labels-idx1-ubyte/train-labels-idx1-ubyte")
    test_img = join(data_dir, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
    test_lbl = join(data_dir, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

    mnist_loader = MnistDataloader(train_img, train_lbl, test_img, test_lbl)
    (x_train, y_train), (x_test, y_test) = mnist_loader.load_data()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNISTDataset(x_train, y_train, transform=transform)
    test_dataset = MNISTDataset(x_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = create_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Min pixel: {images.min():.4f}, Max pixel: {images.max():.4f}")
    print(f"Label example: {labels[:10].tolist()}")
