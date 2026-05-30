# Vehicle-Signal-Processing-Soccer

멀티 카메라 축구 영상에서 선수 궤적과 포즈 신호를 이용해 카메라 간 시간 오프셋을 추정하는 프로젝트입니다.

## Project Overview

이 프로젝트의 목표는 서로 다른 카메라로 촬영된 축구 영상들을 공통 월드 좌표계와 시간축 위에서 맞추는 것입니다. 현재 구현은 전체 파이프라인 중 앞단인 데이터 로딩, 사람 검출, 월드 좌표 투영, 카메라별 트래킹까지를 중심으로 구성되어 있습니다.

전체 흐름은 다음과 같습니다.

1. ISSIA-Soccer 영상과 annotation을 dataloader로 읽습니다.
2. YOLO를 사용해 각 카메라 프레임에서 사람만 검출합니다.
3. reference image를 클릭 기반으로 캘리브레이션해 image 좌표를 축구장 world 좌표로 투영합니다.
4. world 좌표계의 선수 위치 `px, py`를 이용해 카메라별 Kalman Filter tracking을 수행합니다.
5. 이후 카메라 간 track matching, MMPose 기반 pose estimation, pose vector NCC를 이용한 temporal calibration으로 확장합니다.

현재 주요 실행 진입점:

```bash
python main.py                 # YOLO -> projection -> Kalman tracking
python -m tools.calibration    # reference BMP 클릭 기반 homography calibration
python -m tools.debug_dataset  # ISSIA dataloader sanity check
python -m tools.debug_video    # temporal offset debug video 생성
```

이 레포는 로컬 git repository를 그대로 유지한 상태에서, `VSP_Soccer_ws`에 준비된 Docker 환경 안에서 개발/실행하는 것을 기준으로 합니다.

## Development Environment

Docker 환경은 GPU 사용을 전제로 구성되어 있습니다.

- Ubuntu 20.04
- CUDA 11.7.1 + cuDNN 8
- Python 3
- PyTorch 2.0.1 + CUDA 11.7
- Ultralytics YOLO
- MMPose + MMDetection
- OpenCV
- Common scientific Python packages
- X11 forwarding for GUI/OpenCV display

> CPU-only 환경이 아니라 NVIDIA GPU/CUDA 기반 환경입니다.

## Prerequisites

호스트 PC에는 아래 항목이 준비되어 있어야 합니다.

- NVIDIA GPU driver
- Docker
- NVIDIA Container Toolkit
- X11 display access

GPU가 Docker 안에서 보이는지 확인하려면:

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi
```

## Workspace Layout

현재 로컬 워크스페이스 구조는 다음과 같습니다.

```text
VSP_Soccer_ws/
├── Dockerfile
├── docker.sh
├── requirements.txt
└── Vehicle-Signal-Processing-Soccer/
    ├── config.py
    ├── main.py
    ├── data/
    ├── detectors/
    ├── pipeline/
    ├── projection/
    ├── tracking/
    ├── tools/
    └── weights/
```

`docker.sh`는 로컬의 전체 `VSP_Soccer_ws`를 컨테이너에 마운트합니다.

Local path:

```text
<your VSP_Soccer_ws path>
```

Container path:

```text
/workspace/VSP_Soccer_ws
```

따라서 컨테이너 안에서 이 레포의 위치는:

```text
/workspace/VSP_Soccer_ws/Vehicle-Signal-Processing-Soccer
```

## Build Docker Image

Docker image는 `VSP_Soccer_ws/`에서 빌드합니다.

```bash
cd <your VSP_Soccer_ws path>
sudo docker build -t vspsoccer-image:latest .
```

빌드 과정에서 `requirements.txt`가 이미지 안으로 복사되고, Python 패키지들이 함께 설치됩니다.

## Run Docker Container

컨테이너 실행:

```bash
cd <your VSP_Soccer_ws path>
chmod +x docker.sh
./docker.sh
```

`docker.sh`는 다음 설정으로 컨테이너를 실행합니다.

- container name: `VSPSoccer-container`
- image name: `vspsoccer-image:latest`
- GPU: `--gpus "device=0"`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- ISSIA dataset root: `ISSIA_SOCCER_ROOT=/ssd/ISSIA-Soccer`
- network: host network
- IPC: host IPC for PyTorch dataloader/shared memory
- device access: `/dev`
- display forwarding: `/tmp/.X11-unix`, `DISPLAY`
- workspace mount: `<your VSP_Soccer_ws path>` -> `/workspace/VSP_Soccer_ws`
- dataset/SSD mount: `<your ISSIA parent data path>` -> `/ssd`

## Inside The Container

컨테이너가 실행되면 프로젝트 레포로 이동합니다.

```bash
cd /workspace/VSP_Soccer_ws/Vehicle-Signal-Processing-Soccer
```

GPU/PyTorch 연결 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

YOLO/MMPose import 확인:

```bash
python -c "import ultralytics, mmpose, mmdet, mmcv; print(ultralytics.__version__); print(mmpose.__version__); print(mmdet.__version__); print(mmcv.__version__)"
```

OpenCV import 확인:

```bash
python -c "import cv2; print(cv2.__version__)"
```

## Project Layout

레포 내부 코드는 아래처럼 나눕니다.

```text
Vehicle-Signal-Processing-Soccer/
├── config.py              # 경로, 모델, dataloader, tracking 기본값
├── main.py                # YOLO -> projection -> Kalman tracking 실행
├── data/                  # ISSIA dataset/dataloader
├── detectors/             # YOLO detector wrapper
├── pipeline/              # end-to-end pipeline glue
├── projection/            # image 좌표 -> world 좌표 projection
├── tracking/              # world-frame Kalman filter tracker
├── tools/                 # calibration/debug helper scripts
├── weights/               # YOLO/MMPose 등 로컬 가중치
└── debug_outputs/         # 실행 결과물, calibration 결과물
```

가중치는 기본적으로 `weights/` 아래에 둡니다. 현재 YOLO 기본 경로는:

```text
weights/yolo26x.pt
```

다른 weight를 쓰려면 `config.py`의 `YOLO_MODEL_PATH`를 바꾸거나 환경변수로 지정합니다.

```bash
YOLO_MODEL_PATH=/path/to/model.pt python main.py
```

주요 실행 명령:

```bash
python main.py
python -m tools.calibration
python -m tools.debug_dataset
python -m tools.debug_video
```

## Docker Files

Docker 관련 파일은 git repository 바깥의 `VSP_Soccer_ws/`에 있습니다.

```text
<your VSP_Soccer_ws path>/Dockerfile
<your VSP_Soccer_ws path>/docker.sh
<your VSP_Soccer_ws path>/requirements.txt
```

### Dockerfile Summary

현재 Dockerfile은 다음 내용을 포함합니다.

- `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04` 기반 이미지 사용
- 기본 개발 도구 설치: `git`, `build-essential`, `cmake`, `ninja-build`, `python3-pip` 등
- PyTorch CUDA 11.7 버전 설치
- `requirements.txt` 기반 Python package 설치
- Ultralytics YOLO 설치
- MMPose 실행을 위한 OpenMMLab package 설치: `mmengine`, `mmcv`, `mmdet`, `mmpose`
- OpenCV 및 GUI 실행에 필요한 dependency 설치

### docker.sh Summary

현재 실행 스크립트는 GPU 0번을 사용하는 컨테이너를 띄우고, 로컬 workspace와 SSD 경로를 컨테이너 안으로 연결합니다.

```bash
./docker.sh
```

컨테이너 이름이 이미 존재한다는 에러가 나면 기존 컨테이너를 삭제한 뒤 다시 실행합니다.

```bash
sudo docker rm VSPSoccer-container
./docker.sh
```

## Notes

- `Dockerfile`, `docker.sh`, `requirements.txt`는 `Vehicle-Signal-Processing-Soccer` 레포 안이 아니라 상위 workspace인 `VSP_Soccer_ws/`에 있습니다.
- 컨테이너 실행 전에 Docker image 이름이 `vspsoccer-image:latest`로 빌드되어 있어야 합니다.
- `docker.sh`는 GPU 0번만 사용하도록 설정되어 있습니다. 다른 GPU를 사용하려면 `--gpus "device=0"`과 `CUDA_VISIBLE_DEVICES=0` 값을 수정하면 됩니다.
- OpenCV GUI 창을 사용하려면 호스트의 X11 권한이 필요하며, `docker.sh`에서 `xhost +local:docker`를 실행합니다.

## Full Dockerfile

아래는 이 프로젝트에서 실제로 사용한 Dockerfile입니다. `requirements.txt`는 Dockerfile과 같은 `VSP_Soccer_ws/` 디렉토리에 있어야 합니다.

```dockerfile
# NVIDIA CUDA 기반 이미지 사용
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables to prevent tzdata from prompting for geographic area
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 필수 패키지 및 OpenCV/GUI dependency 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    curl \
    git \
    build-essential \
    ninja-build \
    unzip \
    python3-pip \
    python3-dev \
    nano \
    python-is-python3 \
    cmake \
    pkg-config \
    ffmpeg \
    libglib2.0-0 \
    libopencv-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libdbus-1-3 \
    libx11-xcb1 \
    libxcb1 \
    libxcb-glx0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxcb-cursor0 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch CUDA 11.7 설치
RUN python -m pip install "networkx<3.0"
RUN python -m pip install \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    torchaudio==2.0.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# requirements.txt 복사
COPY requirements.txt /home/requirements.txt

# 필요한 패키지 설치
RUN python -m pip install -r /home/requirements.txt

# Ultralytics YOLO 설치
RUN python -m pip install "ultralytics==8.4.53"

# MMPose/OpenMMLab 설치
# mmcv는 PyTorch 2.0.x + CUDA 11.7 prebuilt wheel을 명시해서 source build를 피한다.
RUN python -m pip install "openmim==0.3.9" \
    && python -m pip install "mmengine==0.10.7" \
    && python -m pip install "mmcv==2.1.0" \
        -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html \
    && mim install "mmdet==3.2.0" \
    && mim install "mmpose==1.3.2" \
    && rm -rf /root/.cache/pip /root/.cache/mim

# 주요 패키지 import 확인
RUN python -c "import torch, ultralytics, mmcv, mmengine, mmdet, mmpose; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('ultralytics', ultralytics.__version__); print('mmcv', mmcv.__version__); print('mmengine', mmengine.__version__); print('mmdet', mmdet.__version__); print('mmpose', mmpose.__version__)"

# 작업 디렉토리 설정
WORKDIR /home
```

## Full docker.sh

아래는 이 프로젝트에서 실제로 사용한 실행 스크립트입니다. 로컬 경로는 본인 환경에 맞게 `<your ...>` 부분만 바꿔서 사용합니다.

```bash
#!/bin/bash
# sudo docker build -t vspsoccer-image:latest .
xhost +local:docker
xhost +local:root
sudo docker run --name VSPSoccer-container -it \
  --privileged \
  --gpus "device=0" \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e ISSIA_SOCCER_ROOT=/ssd/ISSIA-Soccer \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v /dev:/dev \
  -v <your VSP_Soccer_ws path>:/workspace/VSP_Soccer_ws \
  -v "<your ISSIA parent data path>:/ssd" \
  vspsoccer-image:latest
```
