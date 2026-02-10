from ultralytics import YOLO
import time
import cv2
import numpy as np
from collections import defaultdict

model = YOLO("best.pt")
# ...existing code...

if __name__ == "__main__":
    # ROI (adjust to your video/frame resolution): (x1, y1, x2, y2)
    ROI = (300, 200, 1900, 1080)

    counted = defaultdict(set)  # class_idx -> set(track_ids)
    counts = defaultdict(int)   # class_idx -> count

    start = time.perf_counter()
    results = model.track(
        source='Test Data/Test3Short.mp4',
        tracker='bytetrack.yaml',
        persist=True,
        save=True,
        save_txt=True,
        line_width = 1,
        show = True
    )

    for frame_idx, r in enumerate(results):
        img = getattr(r, "orig_img", None)
        if img is None:
            continue

        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            cv2.rectangle(img, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (0,255,0), 2)
            continue

        # Convert box attributes to numpy arrays
        xyxy = boxes.xyxy.cpu().numpy()       # (N,4)
        cls = boxes.cls.cpu().numpy().astype(int)  # (N,)
        try:
            ids = boxes.id.cpu().numpy().astype(int)  # (N,)
        except Exception:
            ids = np.arange(len(xyxy)) + frame_idx * 100000  # fallback unique tokens

        for (x1, y1, x2, y2), c, tid in zip(xyxy, cls, ids):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]:
                if tid not in counted[c]:
                    counted[c].add(tid)
                    counts[c] += 1

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
            cv2.putText(img, f"id:{int(tid)} cls:{int(c)}", (int(x1), int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Draw ROI and counts
        cv2.rectangle(img, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (0,255,0), 2)
        y = 30
        for cl, cnt in counts.items():
            cv2.putText(img, f"class {cl}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            y += 30

        cv2.imshow("tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    end = time.perf_counter()
    print("Counts:", dict(counts))
    print(f"Time: {end - start: .2f}")