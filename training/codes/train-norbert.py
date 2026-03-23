import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import PIL.Image
import cv2
import numpy as np
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# ===== KONFIGURACJA =====
DATASET_PATH = r'DATASET-lewo'
BATCH_SIZE   = 8
EPOCHS       = 70
SAVE_PATH    = 'road_following_model_left_laptop.pth'
# ========================


# ===== DATASET =====
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, random_hflip=False):
        super(XYDataset, self).__init__()
        self.directory    = directory
        self.transform    = transform
        self.random_hflip = random_hflip
        self.refresh()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann    = self.annotations[idx]
        image  = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)

        # (opcjonalnie lepiej)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image  = PIL.Image.fromarray(image)

        width  = image.width
        height = image.height

        if self.transform is not None:
            image = self.transform(image)

        # normalizacja do [-1, 1]
        x = 2.0 * (ann['x'] / width  - 0.5)
        y = 2.0 * (ann['y'] / height - 0.5)

        if self.random_hflip and (np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            x = -x

        return image, torch.Tensor([x, y])

    def _parse(self, path):
        basename = os.path.basename(path)
        items    = basename.split('_')
        return int(items[0]), int(items[1])

    def refresh(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, '*.jpg')):
            x, y = self._parse(image_path)
            self.annotations.append({
                'image_path': image_path,
                'x': x,
                'y': y
            })
        print(f"Znaleziono: {len(self.annotations)} zdjęć")


# ===== MODEL =====
class RoadFollowingModel(nn.Module):
    def __init__(self):
        super(RoadFollowingModel, self).__init__()

        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        x = torch.tanh(x)
        return x


# ===== MAIN =====
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Urządzenie: {device}")

    TRANSFORMS = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    dataset = XYDataset(DATASET_PATH, TRANSFORMS, random_hflip=True)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = RoadFollowingModel().to(device)

    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    loss_history = []
    epochs_left = EPOCHS

    while epochs_left > 0:
        i = 0
        sum_loss = 0.0

        for images, xy in train_loader:
            images = images.to(device)
            xy     = xy.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = torch.mean((outputs - xy) ** 2)

            loss.backward()
            optimizer.step()

            i        += len(xy)
            sum_loss += float(loss)

            print(f"\r  loss: {sum_loss/i:.6f}  progress: {i/len(dataset)*100:.1f}%", end='')

        epochs_left -= 1

        epoch_loss = sum_loss / i
        loss_history.append(epoch_loss)

        print(f"\nEpoch {EPOCHS - epochs_left}/{EPOCHS} | Loss: {epoch_loss:.6f}")

    model.eval()

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nZapisano: {SAVE_PATH}")

    # ===== WYKRES =====
    plt.figure()
    plt.plot(loss_history, marker='o')
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.title("Loss w czasie treningu")
    plt.grid()
    plt.show()