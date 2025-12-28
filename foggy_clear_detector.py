import torch
import torch.nn as nn

#  StrongerMLP 모델 정의
class StrongerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

#  안개/맑음 판단기
class FoggyClearDetector:
    def __init__(self, model_path, device, threshold=0.10):
        self.device = device
        self.threshold = threshold

        self.model = StrongerMLP().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict(self, features_tensor):
        with torch.no_grad():
            pred = self.model(features_tensor)
            probs = torch.softmax(pred, dim=1)[0]
            foggy_prob = probs[1].item()
            pred_label = int(foggy_prob >= self.threshold)
            confidence = foggy_prob if pred_label == 1 else probs[0].item()
        foggy_clear_label = "FOGGY" if pred_label == 1 else "CLEAR"
        return foggy_clear_label, confidence

