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
# 1.1. Chỉ dùng một số nhãn (tùy chọn)
# ============================
# Hãy điền các nhãn bạn muốn giữ lại trong danh sách dưới đây.
# Nếu để trống (mặc định), mô hình sẽ dùng tất cả nhãn của bộ weights.
allowed_labels = [
    "wrestling", "punching bag", "punching person", "kissing", "hugging", "exercising arm"
]

# Chuyển tên nhãn -> index trong danh sách gốc
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
    probs = F.softmax(outputs, dim=1)[0]  # (num_classes,)

    if allowed_indices is not None:
        subset_probs = probs[allowed_indices]  # (len(allowed_indices),)
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

# ============================
# 6. Kết quả
# ============================
print("Predicted class:", pred_label, f"(id={pred_id})")
print("Confidence:", confidence)

# In thêm Top-k trong tập nhãn đã lọc (nếu có)
if allowed_indices is not None:
    k = min(5, len(allowed_indices))
    top_vals, top_idx = torch.topk(subset_probs, k)
    print("Top-k (trong các nhãn đã lọc):")
    for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
        print(f"- {labels[allowed_indices[idx]]}: {v:.4f}")