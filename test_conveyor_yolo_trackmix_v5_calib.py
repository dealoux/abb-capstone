from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
import math

# ---------------- Config ----------------
VIDEO_PATH = "vid_ex1.mp4"
MODEL_PATH = "yolov8n.pt"
IMGSZ = 640
CONF = 0.10
IOU_YOLO = 0.5
MAX_DET = 300
DETECT_EVERY = 3
DELAY = 1

MIN_W, MIN_H = 24, 24
AREA_MIN = 1200
AREA_MAX = 20_000_000
ASPECT_MIN, ASPECT_MAX = 0.25, 3.5
RECT_MIN_EXTENT = 0.45
SOLIDITY_MIN = 0.75

BLUR_KSIZE = 3
CLOSE_K = (17, 9)
OPEN_VERT_K = (3, 9)
SAT_MIN = 35

lk_win = (21, 21)
lk_term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

WINDOW_NAME = "FAST v3: YOLO+CV+OF | TR-origin coords (cm after calib)"
SHOW_FPS = True

# ------- Robot marker config -------
ROBOT_BOX_SIZE = 60
ROBOT_MARGIN_X = 20
ROBOT_MARGIN_Y = 20
ROBOT_COLOR = (0, 0, 255)  # RED
ROBOT_TEXT = "ROBOT"
ROBOT_TEXT_CLR = (255, 255, 255)

# ------- Calibration state -------
px_per_cm = None  # computed after picking 2 points = px / cm
calib_pts = []  # two points clicked
calib_mode = False  # press 'c' to start capture


# ---------------- Helpers ----------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms_merge(bboxes, iou_thr=0.45):
    if not bboxes:
        return []
    boxes = np.array(bboxes, dtype=float)
    order = boxes[:, 4].argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if len(rest) == 0:
            break
        ious = np.array([iou_xyxy(boxes[i, :4], boxes[j, :4]) for j in rest])
        order = rest[ious < iou_thr]
    out = boxes[keep].astype(int)
    return [list(map(int, b[:4])) for b in out]


def corners_from_xyxy(b):
    x1, y1, x2, y2 = map(float, b)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def xyxy_from_corners(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    return [int(x.min()), int(y.min()), int(x.max()), int(y.max())]


def detect_boxes_cv_fast(frame):
    h, w = frame.shape[:2]
    step = h // 3
    lanes = [(0, i * step, w, h if i == 2 else (i + 1) * step) for i in range(3)]

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    _, _, B = cv2.split(lab)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)

    boxes = []
    for x0, y0, x1, y1 in lanes:
        roiB = B[y0:y1, x0:x1]
        roiS = S[y0:y1, x0:x1]

        roiB = cv2.GaussianBlur(roiB, (BLUR_KSIZE, BLUR_KSIZE), 0)
        _, maskB = cv2.threshold(roiB, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        maskS = cv2.inRange(roiS, SAT_MIN, 255)
        mask = cv2.bitwise_and(maskB, maskS)

        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_K)
        k_openv = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_VERT_K)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_openv, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < AREA_MIN or area > AREA_MAX:
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = rect
            bw, bh = int(rw), int(rh)
            if bw < MIN_W or bh < MIN_H:
                continue

            hull = cv2.convexHull(c)
            solidity = area / (cv2.contourArea(hull) + 1e-6)
            box_pts = cv2.boxPoints(rect).astype(int)
            x, y, wbox, hbox = cv2.boundingRect(box_pts)
            rect_area = wbox * hbox
            extent = area / (rect_area + 1e-6)
            aspect = max(rw, rh) / (min(rw, rh) + 1e-6)

            if solidity < SOLIDITY_MIN:
                continue
            if extent < RECT_MIN_EXTENT:
                continue
            if aspect < 1 / ASPECT_MAX or aspect > ASPECT_MAX:
                continue

            boxes.append([x0 + x, y0 + y, x0 + x + wbox, y0 + y + hbox, 0.55])
    return boxes


# ---------------- Mouse for calibration ----------------
def on_mouse(event, x, y, flags, param):
    global calib_mode, calib_pts, px_per_cm
    if not calib_mode:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        calib_pts.append((x, y))
        if len(calib_pts) == 2:
            # compute pixels per cm using 1 meter = 100 cm
            (x1, y1), (x2, y2) = calib_pts
            dist_px = math.hypot(x2 - x1, y2 - y1)
            if dist_px > 1:
                px_per_cm = dist_px / 100.0
            calib_mode = False  # done
            # keep the 2 points to draw the reference line
            print(f"[CALIB] px_per_cm = {px_per_cm:.3f} (px/cm)")


# ---------------- Init ------------------
cv2.setUseOptimized(True)
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
if device == 0:
    try:
        model.to("cuda")
        model.model.half()
        model.fuse()
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Failed to open video: {VIDEO_PATH}")
    raise SystemExit
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

prev_gray = None
tracks = {}
next_id = 0
frame_idx = 0
t_last = time.time()
fps_disp = 0.0

# ---------------- Main loop -------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video ended.")
        break
    frame_idx += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = frame.shape[:2]

    must_detect = (frame_idx % DETECT_EVERY == 1) or (len(tracks) == 0)

    if must_detect:
        yolo_boxes = []
        res = model.predict(
            frame,
            device=(0 if torch.cuda.is_available() else "cpu"),
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU_YOLO,
            max_det=MAX_DET,
            agnostic_nms=True,
            verbose=False,
        )[0]
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
                continue
            yolo_boxes.append([x1, y1, x2, y2, float(box.conf[0])])

        cv_boxes = detect_boxes_cv_fast(frame)
        merged = nms_merge(yolo_boxes + cv_boxes, iou_thr=0.45)

        used = set()
        for tid, t in list(tracks.items()):
            best_iou, best_j = 0.0, -1
            for j, d in enumerate(merged):
                if j in used:
                    continue
                iou = iou_xyxy(t["box"], d)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou > 0.2:
                d = merged[best_j]
                used.add(best_j)
                t["box"] = d
                t["corners"] = corners_from_xyxy(d)
            else:
                tracks.pop(tid, None)

        for j, d in enumerate(merged):
            if j in used:
                continue
            tracks[next_id] = {
                "box": d,
                "corners": corners_from_xyxy(d),
                "color": (0, 255, 0),
            }
            next_id += 1

    elif prev_gray is not None and len(tracks):
        pts_list, ids = [], []
        for tid, t in tracks.items():
            pts_list.append(t["corners"])
            ids.append(tid)
        pts_prev = np.concatenate(pts_list, axis=0).reshape(-1, 1, 2)

        pts_next, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            pts_prev,
            None,
            winSize=lk_win,
            maxLevel=3,
            criteria=lk_term,
        )
        if pts_next is not None:
            pts_next = pts_next.reshape(-1, 2)
            st = st.reshape(-1)
            k = 0
            for tid in ids:
                pts_track = []
                valid = 0
                for _ in range(4):
                    if st[k] == 1:
                        pts_track.append(pts_next[k])
                        valid += 1
                    k += 1
                if valid >= 2:
                    pts_track = np.array(pts_track, dtype=np.float32)
                    x1, y1, x2, y2 = xyxy_from_corners(pts_track)
                    if (x2 - x1) >= MIN_W and (y2 - y1) >= MIN_H:
                        tracks[tid]["box"] = [x1, y1, x2, y2]
                        tracks[tid]["corners"] = corners_from_xyxy([x1, y1, x2, y2])
                else:
                    tracks.pop(tid, None)

    # ----- Draw tracked boxes -----
    for tid, t in list(tracks.items()):
        x1, y1, x2, y2 = t["box"]
        if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
            tracks.pop(tid, None)
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # TR-origin (pixel)
        x_tr_px = cx - W
        y_tr_px = -cy

        # Convert to cm if calibrated
        if px_per_cm and px_per_cm > 0:
            x_tr_cm = int(round(x_tr_px / px_per_cm))
            y_tr_cm = int(round(y_tr_px / px_per_cm))
            coord_text = f"({x_tr_cm},{y_tr_cm}) cm"
        else:
            coord_text = f"({x_tr_px},{y_tr_px})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), t["color"], 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            coord_text,
            (x1, max(14, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    # ----- Draw Robot marker (always on top) -----
    rb_x2 = W - ROBOT_MARGIN_X
    rb_x1 = rb_x2 - ROBOT_BOX_SIZE
    rb_y1 = ROBOT_MARGIN_Y
    rb_y2 = rb_y1 + ROBOT_BOX_SIZE
    cv2.rectangle(frame, (rb_x1, rb_y1), (rb_x2, rb_y2), ROBOT_COLOR, -1)
    (tw, th), _ = cv2.getTextSize(ROBOT_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    tx = rb_x1 + (ROBOT_BOX_SIZE - tw) // 2
    ty = rb_y2 + th + 10
    cv2.putText(
        frame, ROBOT_TEXT, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ROBOT_TEXT_CLR, 3
    )

    # ----- Draw calibration aids -----
    # 1) If in capture mode, show instruction
    if calib_mode:
        cv2.putText(
            frame,
            "Click 2 points = 1 meter (press 'r' to reset)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    # 2) Draw chosen points / line
    if len(calib_pts) >= 1:
        cv2.circle(frame, calib_pts[0], 6, (0, 0, 255), -1)
    if len(calib_pts) == 2:
        cv2.circle(frame, calib_pts[1], 6, (0, 0, 255), -1)
        cv2.line(frame, calib_pts[0], calib_pts[1], (0, 0, 255), 3)
        # label "100 cm"
        midx = (calib_pts[0][0] + calib_pts[1][0]) // 2
        midy = (calib_pts[0][1] + calib_pts[1][1]) // 2
        cv2.putText(
            frame,
            "100 cm",
            (midx + 8, midy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # 3) Optional top scale bar with 100 cm if calibrated
    if px_per_cm and px_per_cm > 0:
        length_px = int(100 * px_per_cm)
        length_px = min(length_px, W - 80)  # clip if too long for window
        sx = (W - length_px) // 2
        sy = 30
        cv2.line(frame, (sx, sy), (sx + length_px, sy), (0, 0, 255), 4)
        cv2.line(frame, (sx, sy - 8), (sx, sy + 8), (0, 0, 255), 3)
        cv2.line(
            frame, (sx + length_px, sy - 8), (sx + length_px, sy + 8), (0, 0, 255), 3
        )
        cv2.putText(
            frame,
            "100 cm",
            (sx + length_px // 2 - 50, sy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # FPS
    if SHOW_FPS:
        now = time.time()
        dt = now - t_last
        if dt > 0:
            fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / dt) if fps_disp > 0 else (1.0 / dt)
        t_last = now
        cv2.putText(
            frame,
            f"FPS: {fps_disp:.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    cv2.imshow(WINDOW_NAME, frame)
    prev_gray = gray

    key = cv2.waitKey(DELAY) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("p"):
        while True:
            k2 = cv2.waitKey(0) & 0xFF
            if k2 in (ord("p"), ord("q")):
                if k2 == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit
                break
    elif key == ord("c"):  # start calibration
        calib_mode = True
        calib_pts = []
        print("[CALIB] Click two points that are 1 meter apart.")
    elif key == ord("r"):  # reset calibration
        px_per_cm = None
        calib_pts = []
        calib_mode = False
        print("[CALIB] Reset.")

cap.release()
cv2.destroyAllWindows()
