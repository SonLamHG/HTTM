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
# 1.1. Chỉ dùng một số nhãn (tùy chọn)
# ============================
# Điền các nhãn bạn muốn giữ lại tại đây. Để trống = dùng tất cả nhãn.
# Ví dụ: ["wrestling", "boxing", "punching bag"]
allowed_labels = [
    "wrestling", "punching bag", "punching person", "kissing", "hugging", "exercising arm"
]

if len(allowed_labels) > 0:
    allowed_indices = []
    invalid = []
    for name in allowed_labels:
        if name in labels:
            allowed_indices.append(labels.index(name))
        else:
            invalid.append(name)
    if invalid:
        print(f"⚠️ Các nhãn không tồn tại và sẽ bị bỏ qua: {invalid}")
    if not allowed_indices:
        raise ValueError("❌ Không còn nhãn hợp lệ sau khi lọc. Hãy cập nhật 'allowed_labels'.")
else:
    allowed_indices = None  # None = dùng tất cả nhãn

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
cap = cv2.VideoCapture("video1.mp4")
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
    probs = F.softmax(outputs, dim=1)[0]  # (num_classes,)

    if allowed_indices is not None:
        subset_probs = probs[allowed_indices]
        if subset_probs.numel() == 0:
            raise ValueError("❌ Không có xác suất nào trong tập nhãn đã lọc.")
        subset_best_idx = torch.argmax(subset_probs).item()
        pred_id = allowed_indices[subset_best_idx]
        pred_label = labels[pred_id]
        confidence = probs[pred_id].item()
    else:
        pred_id = torch.argmax(probs).item()
        pred_label = labels[pred_id]
        confidence = probs[pred_id].item()

print("Predicted class:", pred_label, f"(id={pred_id})")
print("Confidence:", confidence)

if allowed_indices is not None:
    k = min(5, len(allowed_indices))
    top_vals, top_idx = torch.topk(subset_probs, k)
    print("Top-k (trong các nhãn đã lọc):")
    for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
        print(f"- {labels[allowed_indices[idx]]}: {v:.4f}")
