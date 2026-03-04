"""
server.py - FastMCP + YOLO11 inference MCP server (B안 강화판)

✅ 목적
- Agent Builder가 GPT 비전으로 "추정"하지 못하게, MCP(YOLO) 결과를
  다음 에이전트/워크플로우가 그대로 파싱 가능한 JSON으로 반환
- tool 호출 여부를 서버 로그로 즉시 확인 가능
- 결과 변조/재서술 여부를 감지할 수 있도록 source/nonce 포함

✅ 설치(권장: 서버/컨테이너)
  pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
  pip install --no-cache-dir opencv-python-headless ultralytics fastmcp requests

(필요 시, Ubuntu/Debian)
  sudo apt-get update
  sudo apt-get install -y libgl1 libglib2.0-0

✅ 환경변수
- WEIGHTS=./best.pt
- DEVICE=cpu   (GPU면 "0")
- PORT=8001
- CONF=0.25
- IOU=0.45

✅ 반환(JSON-only)
{
  "source": "yolo11_mcp",
  "nonce": "<uuid>",
  "weights": "<abs path>",
  "device": "cpu|0",
  "top_species": str|null,
  "top_confidence": float|null,
  "species_counts": { "<species>": int, ... },
  "detections": [
    { "species": str, "confidence": float, "bbox_xyxy":[x1,y1,x2,y2] }, ...
  ],
  "error": str? (있을 수도 있음)
}
"""

from __future__ import annotations

import os
import uuid
import base64
import tempfile
from typing import Any, Dict, Optional

import requests
from fastmcp import FastMCP


# --------------------------------------------
# ✅ 부트 진단: cv2 / ultralytics import 문제를 빨리 잡기
# --------------------------------------------
def _boot_diagnostics() -> None:
    pyver = f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    print(f"[BOOT] Python: {pyver}")
    print(f"[BOOT] WEIGHTS={os.getenv('WEIGHTS', '(default)')}, DEVICE={os.getenv('DEVICE', 'cpu')}")

    try:
        import cv2  # noqa: F401
        print("[BOOT] cv2 import: OK")
    except Exception as e:
        print("[BOOT] cv2 import: FAIL")
        print(f"[BOOT] cv2 error: {repr(e)}")
        raise

    try:
        from ultralytics import YOLO  # noqa: F401
        print("[BOOT] ultralytics import: OK")
    except Exception as e:
        print("[BOOT] ultralytics import: FAIL")
        print(f"[BOOT] ultralytics error: {repr(e)}")
        raise


_boot_diagnostics()

# --------------------------------------------
# ✅ YOLO 로드 (서버 시작 시 1회)
# --------------------------------------------
from ultralytics import YOLO  # noqa: E402

mcp = FastMCP("YOLO11 MCP Server")

WEIGHTS_PATH = os.getenv("WEIGHTS", "./best.pt")
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" 또는 "0"(GPU)

CONF_DEFAULT = float(os.getenv("CONF", "0.25"))
IOU_DEFAULT = float(os.getenv("IOU", "0.45"))

WEIGHTS_ABS = os.path.abspath(WEIGHTS_PATH)

print(f"[BOOT] Loading YOLO weights: {WEIGHTS_PATH}")
print(f"[BOOT] WEIGHTS abs: {WEIGHTS_ABS}")
model = YOLO(WEIGHTS_PATH)
print("[BOOT] Model loaded: OK")


# --------------------------------------------
# ✅ class_id -> 종명 (사용자 제공 매핑)
# --------------------------------------------
SPECIES_MAP = {
    0: "Chelydra serpentina",
    1: "Macrochelys temminckii",
    2: "Mauremys sinensis",
    3: "Pseudemys concinna",
    4: "Pseudemys nelsoni",
    5: "Trachemys scripta",
}


# --------------------------------------------
# 유틸: 입력(base64/URL) -> 임시 파일 저장
# --------------------------------------------
def _save_base64_to_tempfile(image_b64: str, suffix: str = ".jpg") -> str:
    # "data:image/jpeg;base64,..." 형태도 처리
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    img_bytes = base64.b64decode(image_b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(img_bytes)
    return path


def _download_url_to_tempfile(url: str, timeout: int = 25, suffix: str = ".jpg") -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return path


# --------------------------------------------
# ✅ Ultralytics 결과 -> "다음 에이전트가 바로 소비" 가능한 JSON
# --------------------------------------------
def _results_to_json(results) -> Dict[str, Any]:
    """
    JSON-only response for downstream agent/workflow.
    Includes 'source' and 'nonce' to detect non-tool answers or transformations.
    """
    out: Dict[str, Any] = {
        "source": "yolo11_mcp",
        "nonce": str(uuid.uuid4()),
        "weights": WEIGHTS_ABS,
        "device": DEVICE,
        "top_species": None,
        "top_confidence": None,
        "species_counts": {},
        "detections": [],
    }

    if not results or len(results) == 0:
        return out

    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return out

    xyxy = r0.boxes.xyxy.cpu().numpy()
    conf = r0.boxes.conf.cpu().numpy()
    cls = r0.boxes.cls.cpu().numpy().astype(int)

    # ✅ top-1
    best_idx = int(conf.argmax())
    best_class_id = int(cls[best_idx])
    best_species = SPECIES_MAP.get(best_class_id, f"unknown_{best_class_id}")
    out["top_species"] = best_species
    out["top_confidence"] = float(conf[best_idx])

    # ✅ all detections + species counts
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        confidence = float(conf[i])
        class_id = int(cls[i])

        species = SPECIES_MAP.get(class_id, f"unknown_{class_id}")

        out["species_counts"][species] = out["species_counts"].get(species, 0) + 1

        out["detections"].append(
            {
                "species": species,
                "confidence": confidence,
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )

    return out


# --------------------------------------------
# ✅ MCP Tool: YOLO11 추론
# --------------------------------------------
@mcp.tool
def yolo11_predict(
    image_b64: Optional[str] = None,
    image_url: Optional[str] = None,
    conf: float = CONF_DEFAULT,
    iou: float = IOU_DEFAULT,
    max_det: int = 300,
    imgsz: int = 640,
) -> Dict[str, Any]:
    """
    입력(둘 중 하나 필수):
      - image_b64: base64 인코딩 이미지 문자열
      - image_url: 이미지 URL

    출력(JSON-only):
      - source, nonce, top_species, top_confidence, species_counts, detections
    """

    # ✅ 서버 로그로 "도구가 실제 호출되는지" 즉시 확인
    print(
        "[CALL] yolo11_predict",
        {
            "has_b64": bool(image_b64),
            "has_url": bool(image_url),
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "max_det": max_det,
            "device": DEVICE,
        },
    )

    if not image_b64 and not image_url:
        return {
            "source": "yolo11_mcp",
            "nonce": str(uuid.uuid4()),
            "weights": WEIGHTS_ABS,
            "device": DEVICE,
            "top_species": None,
            "top_confidence": None,
            "species_counts": {},
            "detections": [],
            "error": "image_b64 또는 image_url 중 하나는 반드시 필요합니다.",
        }

    tmp_path: Optional[str] = None

    try:
        if image_b64:
            tmp_path = _save_base64_to_tempfile(image_b64)
        else:
            tmp_path = _download_url_to_tempfile(image_url)  # type: ignore[arg-type]

        results = model.predict(
            source=tmp_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=DEVICE,
            verbose=False,
        )

        return _results_to_json(results)

    except Exception as e:
        return {
            "source": "yolo11_mcp",
            "nonce": str(uuid.uuid4()),
            "weights": WEIGHTS_ABS,
            "device": DEVICE,
            "top_species": None,
            "top_confidence": None,
            "species_counts": {},
            "detections": [],
            "error": str(e),
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# --------------------------------------------
# 서버 실행
# --------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
        path="/mcp",
    )
