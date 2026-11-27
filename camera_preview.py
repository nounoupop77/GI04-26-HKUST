"""双相机实时预览 (不需要连接任何机械臂/leader arm)。

功能:
  - 同时显示 top / wrist 两路相机画面 (与录制/部署相同的参数: MJPG 640x480 20fps)
  - 实时叠加青色方块的画面坐标 (核对相机对位用)
  - 按 s 保存当前两帧截图到 camera_check/
  - 按 q 退出

用法 (lerobot 目录下):
  python camera_preview.py                # 默认 top=1, wrist=0
  python camera_preview.py --top 0 --wrist 1   # 索引和实际相反时手动指定
"""
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

OUT = Path("camera_check")


def open_cam(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    return cap


def cyan_centroid(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0].astype(int) >= 85) & (hsv[:, :, 0].astype(int) <= 100)
         & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 80))
    ys, xs = np.nonzero(m)
    if len(xs) < 30:
        return None
    return int(xs.mean()), int(ys.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--wrist", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    cams = {"top": open_cam(args.top), "wrist": open_cam(args.wrist)}
    print("按 s 保存截图, 按 q 退出。")
    while True:
        frames = {}
        for name, cap in cams.items():
            ok, frame = cap.read()
            if not ok:
                continue
            c = cyan_centroid(frame)
            if c:
                cv2.putText(frame, f"block=({c[0]},{c[1]})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.circle(frame, c, 12, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "no block", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow(name, frame)
            frames[name] = frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and frames:
            ts = datetime.now().strftime("%H%M%S")
            for name, frame in frames.items():
                p = OUT / f"preview_{name}_{ts}.png"
                cv2.imwrite(str(p), frame)
                print(f"已保存 {p}")
        elif key == ord("q"):
            break
    for cap in cams.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
