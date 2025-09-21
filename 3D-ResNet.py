import torch
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
import torchvision.transforms as T
import cv2
from PIL import Image

# ============================
# 1. Load pretrained model
# ============================
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.eval()

labels = weights.meta["categories"]

# ============================
# 2. Transform cho từng frame
# ============================
frame_transform = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989])
])

# ============================
# 3. Đọc video & lấy 16 frames
# ============================
cap = cv2.VideoCapture("fight_scene.mp4")
frames = []
num_frames = 16

for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = frame_transform(frame)  # (3, H, W)
    frames.append(frame)

cap.release()

if len(frames) < num_frames:
    raise ValueError("❌ Không đủ số lượng frames từ video!")

# ============================
# 4. Gom thành tensor video
# ============================
video_tensor = torch.stack(frames, dim=1).unsqueeze(0)  # (1, 3, T, H, W)
print("Video tensor shape:", video_tensor.shape)

# ============================
# 5. Inference
# ============================
with torch.no_grad():
    outputs = model(video_tensor)  # (1, num_classes)
    probs = F.softmax(outputs, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()

# ============================
# 6. Kết quả
# ============================
print("Predicted class:", pred_id)
print("Confidence:", probs[0][pred_id].item())