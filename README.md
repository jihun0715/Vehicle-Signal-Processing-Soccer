# Vehicle-Signal-Processing-Soccer

Vehicle Signal Processing soccer project workspace.

이 레포는 로컬 git repository를 그대로 유지한 상태에서, `VSP_Soccer_ws`에 준비된 Docker 환경 안에서 개발/실행하는 것을 기준으로 합니다.

## Development Environment

Docker 환경은 GPU 사용을 전제로 구성되어 있습니다.

- Ubuntu 20.04
- CUDA 11.7.1 + cuDNN 8
- Python 3
- PyTorch 2.0.1 + CUDA 11.7
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
    ├── README.md
    └── LICENSE
```

`docker.sh`는 로컬의 전체 `VSP_Soccer_ws`를 컨테이너에 마운트합니다.

Local path:

```text
/home/jihun/Documents/VSP_Soccer_ws
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
cd /home/jihun/Documents/VSP_Soccer_ws
sudo docker build -t vspsoccer-image:latest .
```

빌드 과정에서 `requirements.txt`가 이미지 안으로 복사되고, Python 패키지들이 함께 설치됩니다.

## Run Docker Container

컨테이너 실행:

```bash
cd /home/jihun/Documents/VSP_Soccer_ws
chmod +x docker.sh
./docker.sh
```

`docker.sh`는 다음 설정으로 컨테이너를 실행합니다.

- container name: `VSPSoccer-container`
- image name: `vspsoccer-image:latest`
- GPU: `--gpus "device=0"`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- network: host network
- device access: `/dev`
- display forwarding: `/tmp/.X11-unix`, `DISPLAY`
- workspace mount: `/home/jihun/Documents/VSP_Soccer_ws` -> `/workspace/VSP_Soccer_ws`
- SSD mount: `/media/jihun/Crucial X10` -> `/ssd`

## Inside The Container

컨테이너가 실행되면 프로젝트 레포로 이동합니다.

```bash
cd /workspace/VSP_Soccer_ws/Vehicle-Signal-Processing-Soccer
```

GPU/PyTorch 연결 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

OpenCV import 확인:

```bash
python -c "import cv2; print(cv2.__version__)"
```

## Docker Files

Docker 관련 파일은 git repository 바깥의 `VSP_Soccer_ws/`에 있습니다.

```text
/home/jihun/Documents/VSP_Soccer_ws/Dockerfile
/home/jihun/Documents/VSP_Soccer_ws/docker.sh
/home/jihun/Documents/VSP_Soccer_ws/requirements.txt
```

### Dockerfile Summary

현재 Dockerfile은 다음 내용을 포함합니다.

- `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04` 기반 이미지 사용
- 기본 개발 도구 설치: `git`, `build-essential`, `cmake`, `ninja-build`, `python3-pip` 등
- PyTorch CUDA 11.7 버전 설치
- `requirements.txt` 기반 Python package 설치
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
