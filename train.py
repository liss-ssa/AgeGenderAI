import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from model import UTKFaceDataset, AgeGenderModel
from train_utils import train_epoch, validate

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Инициализация модели
    model = AgeGenderModel().to(device)
    
    # Загрузка данных
    train_dataset = UTKFaceDataset(
        'data/splits/train.csv',
        'data/resized_224',
        transform=get_transforms(train=True)
    )
    
    val_dataset = UTKFaceDataset(
        'data/splits/val.csv',
        'data/resized_224',
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mae, val_acc = validate(model, val_loader, device)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val MAE: {val_mae:.2f}, Gender Acc: {val_acc:.2%}')
        
        # Сохранение модели
        if epoch == 0 or val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')

if __name__ == '__main__':
    main()