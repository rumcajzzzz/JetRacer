import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import PIL.Image
import cv2
import numpy as np
import time

# ===== KONFIGURACJA =====
DATASET_PATH = r'E:\Studia\My\Semestr6\Projekt\Python\DATASET-Left'
BATCH_SIZE   = 128
EPOCHS       = 30
# ========================


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
        image  = PIL.Image.fromarray(image)
        width  = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        x = 2.0 * (ann['x'] / width  - 0.5)
        y = 2.0 * (ann['y'] / height - 0.5)
        if self.random_hflip and np.random.random() > 0.5:  # fix: random_hflip
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Urządzenie: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    TRANSFORMS = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = XYDataset(DATASET_PATH, TRANSFORMS, random_hflip=True)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(512, 2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    epochs_left = EPOCHS
    time_start  = time.time()
    epoch_times = []

    while epochs_left > 0:
        epoch_start = time.time()
        i        = 0
        sum_loss = 0.0

        for images, xy in iter(train_loader):
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

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        epochs_left -= 1

        vram_used  = torch.cuda.memory_reserved() / 1024**2
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"\nEpoch {EPOCHS - epochs_left}/{EPOCHS} | Loss: {sum_loss/i:.6f} | Czas: {epoch_time:.1f}s | VRAM: {vram_used:.0f}MB / {vram_total:.0f}MB")

    final_loss = sum_loss / i
    SAVE_PATH = f'road_following_model_left_{final_loss * 100000:.0f}t_marcel_laptop_2.pth'
    model.eval()
    torch.save(model.state_dict(), SAVE_PATH)

    total = time.time() - time_start
    print(f"\nZapisano: {SAVE_PATH}")
    print(f"Urządzenie:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Epoki:            {EPOCHS}")
    print(f"Łączny czas:      {total:.0f}s  ({total/60:.1f} min)")
    print(f"Średnia na epokę: {sum(epoch_times)/len(epoch_times):.1f}s")