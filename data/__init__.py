"""soccer 데이터셋 관련 public API를 모아 export하는 패키지 파일.

클래스 구현은 `issia_dataset.py`에 있고, 여기서는 `ISSIASoccerFrameDataset`,
`ISSIASoccerSyncDataset`, `ISSIASoccerOffsetSyncDataset`과 dataloader 생성 함수,
annotation/reference image 탐색 함수를 외부에서 짧은 경로로 import할 수 있게 한다.
"""

from .issia_dataset import (
    BALL_LABEL,
    LABEL_NAMES,
    PERSON_LABEL,
    ISSIASoccerFrameDataset,
    ISSIASoccerOffsetSyncDataset,
    ISSIASoccerSyncDataset,
    collate_issia_samples,
    create_issia_dataloader,
    create_issia_offset_dataloader,
    discover_issia_cameras,
    discover_issia_reference_images,
    find_issia_reference_image,
    read_issia_annotations,
)

__all__ = [
    "BALL_LABEL",
    "LABEL_NAMES",
    "PERSON_LABEL",
    "ISSIASoccerFrameDataset",
    "ISSIASoccerOffsetSyncDataset",
    "ISSIASoccerSyncDataset",
    "collate_issia_samples",
    "create_issia_dataloader",
    "create_issia_offset_dataloader",
    "discover_issia_cameras",
    "discover_issia_reference_images",
    "find_issia_reference_image",
    "read_issia_annotations",
]
