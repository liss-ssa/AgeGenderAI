import torch
from model import AgeGenderModel, UTKFaceDataset
from torchvision import transforms
import cv2

def load_model(model_path):
    model = AgeGenderModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        age, gender = model(image)
        return age.item(), gender.item()

if __name__ == '__main__':
    model = load_model('best_model.pth')
    # data/resized_224/1_0_0_20170109193820009.jpg.chip.jpg - путь к тестовому фото
    age, gender_prob = predict('data/resized_224/1_0_0_20170109193820009.jpg.chip.jpg', model)
    print(f'Predicted Age: {age:.1f}')
    print(f'Predicted Gender: {"Male" if gender_prob > 0.5 else "Female"} (prob: {gender_prob:.2%})')