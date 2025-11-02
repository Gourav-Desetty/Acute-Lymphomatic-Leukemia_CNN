from typing import List
import torch
from PIL import Image
from leukemia.exception.exception import CustomException
from leukemia.logging.logger import logging
from torchvision import transforms
from leukemia.nlp.generator.report_generator import LeukemiaReportGenerator
from leukemia.entity.artifact_entity import Predict_image

class Predictor:
    def __init__(self, model, transform=None):
        self.model = model
        self.model.eval()
        self.class_names = ['hem', 'all']
        self.transform = transform or self._get_default_transform()
        self.report_generator = LeukemiaReportGenerator()

    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, model:torch.nn.Module, image_path: str, class_name:List[str]) -> Predict_image:
        try:
            target_image = Image.open(image_path).convert('RGB')
            target_image_transformed = self.transform(target_image)

            with torch.inference_mode():
                target_image_pred = self.model(target_image_transformed.unsqueeze(dim=0))
            target_image_pred_prob = torch.softmax(target_image_pred, dim=1)
            target_image_pred_label = torch.argmax(target_image_pred_prob, dim=1)

            pred_percentage = target_image_pred_prob.max().item() * 100

            predict_image = Predict_image(prediction = class_name[target_image_pred_label], confidence = pred_percentage)

            return predict_image
        except Exception as e:
            raise CustomException(str(e), e)