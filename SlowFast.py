import torch
import torch.nn.functional as F
from pytorchvideo.models.hub import slowfast_r50
import torchvision.transforms as T
from torchvision.models.video import R3D_18_Weights
import cv2
from PIL import Image

# ============================
# 1. Load pretrained SlowFast
# ============================
weights = R3D_18_Weights.DEFAULT
labels = weights.meta["categories"]
model = slowfast_r50(pretrained=True)
model = model.eval()

# ============================
# 2. Transform cho frame
# ============================
frame_transform = T.Compose([
    T.Resize((256, 256)),   # Resize
    T.CenterCrop(224),      # Crop
    T.ToTensor(),
    T.Normalize(mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225])
])

# ============================
# 3. Đọc video & lấy frames
# ============================
cap = cv2.VideoCapture("video2.mp4")
frames = []
num_frames = 32  # SlowFast thường dùng 32 hoặc 64 frames

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
    raise ValueError("❌ Không đủ số lượng frames trong video!")

# ============================
# 4. Gom thành tensor video
# ============================
video_tensor = torch.stack(frames, dim=1).unsqueeze(0)  # (1, 3, T, H, W)
print("Video tensor shape:", video_tensor.shape)

# SlowFast cần 2 đường input: slow pathway + fast pathway
# Slow: lấy sample 1/4 frame rate
slow_pathway = video_tensor[:, :, ::4, :, :]
fast_pathway = video_tensor
inputs = [slow_pathway, fast_pathway]

# ============================
# 5. Inference
# ============================
with torch.no_grad():
    outputs = model(inputs)  # (1, num_classes)
    probs = F.softmax(outputs, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()

print("Predicted class ID:", pred_id)
print("Predicted class label:", labels[pred_id])
print("Confidence:", probs[0][pred_id].item())
