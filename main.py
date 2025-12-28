import cv2
import torch
import numpy as np

from foggy_clear_detector import FoggyClearDetector
from day_night_detector import DayNightDetector
from tetra_brighten import adjust_gamma

# ✅ Feature 추출 함수
def estimate_features_without_aod(frame):
    haze_strength = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = np.std(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    dark_mean = np.mean(np.min(frame, axis=2))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    return [haze_strength, sharpness, brightness, contrast, edge_density, dark_mean, saturation], brightness

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 사용 디바이스: {device}")

# ✅ 판단기 초기화
foggy_clear_detector = FoggyClearDetector(model_path="best_mlp_strong_focal.pth", device=device)
day_night_detector = DayNightDetector()

# ✅ 영상 열기
cap = cv2.VideoCapture('foggy_day_tetra.mp4')
if not cap.isOpened():
    print("❌ 영상을 열 수 없습니다.")
    exit()

# ✅ 실시간 추론
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ 영상 재생 완료!")
        break

    # 1. 원본 프레임으로 밤/낮 판단
    _, brightness_value = estimate_features_without_aod(frame)
    day_night_label = day_night_detector.predict(brightness_value)

    # 2. 밤이면 프레임 밝게 조정
    if day_night_label == "NIGHT":
        frame = adjust_gamma(frame, gamma=2.0)

    # 3. (보정된) 프레임 기준으로 Feature 추출
    features, _ = estimate_features_without_aod(frame)

    # 정규화
    features = np.array([features])
    features[:, 0] /= 1.0
    features[:, 1] /= 500.0
    features[:, 2] /= 255.0
    features[:, 3] /= 128.0
    features[:, 4] /= 1.0
    features[:, 5] /= 255.0
    features[:, 6] /= 255.0

    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    # 4. 안개/맑음 판단
    foggy_clear_label, _ = foggy_clear_detector.predict(features_tensor)

    # 5. 결과 텍스트
    text = f"{foggy_clear_label} / {day_night_label}"

    # 결과 표시
    cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Foggy/Clear + Day/Night Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("✅ 수동 종료!")
        break

cap.release()
cv2.destroyAllWindows()

