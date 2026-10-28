# GFPGAN Real-time Setup Guide

Target: PC mới với GPU RTX 3080 / 4070, Windows 10/11.

---

## A. Cài đặt thủ công (tải + cài trên máy mới)

### A1. Python 3.10
- Tải: https://www.python.org/downloads/release/python-31011/
- Chọn **Windows installer (64-bit)**
- Tick **"Add Python to PATH"** khi cài

### A2. Git
- Tải: https://git-scm.com/download/win
- Cài mặc định

### A3. CUDA Toolkit 12.1 (hoặc 12.x)
- Tải: https://developer.nvidia.com/cuda-12-1-0-download-archive
- Chọn Windows → x86_64 → exe (local)
- Cài mặc định (bao gồm driver nếu cần)
- Sau cài, kiểm tra: `nvcc --version` trong cmd

### A4. NVIDIA Driver
- Nếu chưa có hoặc cũ, tải driver mới nhất cho 3080/4070: https://www.nvidia.com/download/index.aspx
- Kiểm tra: `nvidia-smi` trong cmd

---

## B. Clone repo

```powershell
cd E:\Phat\RandD\AI
git clone https://github.com/TencentARC/GFPGAN.git
cd GFPGAN
```

---

## C. Tạo venv + cài packages (chạy trong cmd/powershell)

```powershell
python -m venv .venv
.venv\Scripts\activate

# PyTorch CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# GFPGAN dependencies
pip install basicsr==1.4.2 facexlib==0.3.0 realesrgan==0.3.0 gfpgan==1.3.8

# Còn thiếu
pip install opencv-python numpy scipy pyyaml tqdm lmdb yapf
```

---

## D. Patch lỗi basicsr (BẮT BUỘC)

File cần sửa (trong venv):
```
.venv\lib\site-packages\basicsr\data\degradations.py
```

**Dòng 8**, tìm:
```python
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```

Sửa thành:
```python
from torchvision.transforms.functional import rgb_to_grayscale
```

> Lý do: `torchvision >= 0.18` đã xóa module `functional_tensor`, chuyển sang `functional`.

---

## E. Download model weights

### E1. GFPGANv1.4 (bắt buộc, 332MB)
```powershell
mkdir experiments\pretrained_models
```
- Tải: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
- Lưu vào: `experiments\pretrained_models\GFPGANv1.4.pth`

### E2. Face detection + parsing (tự động)
Hai file sau sẽ **tự download lần chạy đầu tiên**:
- `gfpgan/weights/detection_Resnet50_Final.pth` (104MB)
- `gfpgan/weights/parsing_parsenet.pth` (81MB)

Nếu máy mới không có internet, tải trước:
- https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
- https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
- Lưu vào: `gfpgan\weights\`

---

## F. Test inference (ảnh tĩnh)

```powershell
python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.4 -s 2 --bg_upsampler none
```

Kết quả trong `results/`:
- `restored_imgs/` — ảnh đã restore
- `cmp/` — so sánh trước/sau

---

## G. Real-time IP Camera

### G1. Cài IP Webcam trên điện thoại
- Google Play: "IP Webcam" (Pavel Khlebovich)
- Mở app → Start server → ghi nhớ URL (vd: `http://192.168.1.22:8080/video`)
- PC và điện thoại cùng WiFi

### G2. Sửa IP trong script
File `realtime_gfpgan.py`, dòng 19:
```python
IP_CAMERA_URL = "http://192.168.1.22:8080/video"
```
Đổi IP cho đúng với điện thoại mới.

### G3. Chạy
```powershell
.venv\Scripts\activate
python realtime_gfpgan.py
```

Controls:
- `+`/`-` : tăng/giảm weight (0=giữ gốc, 1=restore mạnh)
- `r` : bật/tắt restore (để so sánh)
- `q` : thoát

---

## H. Hiệu năng dự kiến

| GPU | FPS (480p, 1 face) | Ghi chú |
|---|---|---|
| GTX 1080 Ti | ~2 FPS | Đã benchmark |
| RTX 3080 | ~4-6 FPS | Ước tính (nhanh gấp ~2-3x) |
| RTX 4070 | ~5-8 FPS | Ước tính (nhanh gấp ~3x) |

> GFPGAN là GAN nặng (76M params), không thể đạt 30 FPS real-time.
> Nếu cần real-time mượt (30+ FPS), dùng BiSeNet face-parsing pipeline thay thế.

---

## Tổng kết cấu trúc thư mục sau setup

```
GFPGAN/
├── .venv/                              # Python 3.10 venv
├── experiments/
│   └── pretrained_models/
│       └── GFPGANv1.4.pth             # 332MB - tải thủ công
├── gfpgan/
│   └── weights/
│       ├── detection_Resnet50_Final.pth # 104MB - tự download
│       └── parsing_parsenet.pth         # 81MB  - tự download
├── inputs/
│   ├── whole_imgs/                      # ảnh test
│   └── cropped_faces/
├── results/                             # output
├── inference_gfpgan.py                  # inference ảnh tĩnh (có sẵn)
├── realtime_gfpgan.py                   # real-time IP camera (copy từ máy cũ)
└── requirements.txt
```

---

## Checklist nhanh

- [ ] Python 3.10 + PATH
- [ ] CUDA Toolkit 12.x
- [ ] NVIDIA driver mới nhất
- [ ] Clone repo
- [ ] Tạo venv + cài packages
- [ ] Patch `degradations.py` dòng 8
- [ ] Download `GFPGANv1.4.pth`
- [ ] Test `inference_gfpgan.py` — chạy OK
- [ ] Copy `realtime_gfpgan.py` + sửa IP
- [ ] Test real-time — thấy cửa sổ side-by-side
