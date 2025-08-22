import cv2 as cv
import numpy as np
import time

# user params
ROI_W, ROI_H = 260, 260          # small box like a mouse sensor
USE_PYRAMID = True               # coarse-to-fine for bigger shifts
LEVELS = 2                       # pyramid levels (0=off)
FX, FY = 520.0, 520.0            # focal length in pixels (from calibration)
ALT_M = 2.0                      # altitude meters (replace with sensor)
SMOOTH_ALPHA = 0.2               # EMA smoothing for output velocity

url = "http://192.168.43.1:8080/video"  # replace with your camera URL
cap = cv.VideoCapture(0)  # or your video file
assert cap.isOpened()

# optional: fix input size for speed
cap.set(cv.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

# Hanning window for phaseCorrelate (reduces edge effects)
def hann(shape):
    
    return cv.createHanningWindow((shape[1], shape[0]), cv.CV_32F)


def roi_from_center(img, w, h):
    H, W = img.shape[:2]
    cx, cy = W//2, H//2
    x0 = max(0, cx - w//2); y0 = max(0, cy - h//2)
    x1 = min(W, x0 + w);    y1 = min(H, y0 + h)
    return x0, y0, x1, y1

ret, prev = cap.read()
prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
x0,y0,x1,y1 = roi_from_center(prev_gray, ROI_W, ROI_H)
prev_roi = prev_gray[y0:y1, x0:x1].astype(np.float32)
hw = hann(prev_roi.shape)

# for timing and smoothing
last_t = time.time()
vx_ema = 0.0; vy_ema = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cur_roi = gray[y0:y1, x0:x1].astype(np.float32)

    # optional pyramid
    scale_dx = 0.0; scale_dy = 0.0
    if USE_PYRAMID:
        prev_pyr = [prev_roi]; cur_pyr = [cur_roi]
        for _ in range(LEVELS):
            prev_pyr.append(cv.pyrDown(prev_pyr[-1]))
            cur_pyr.append(cv.pyrDown(cur_pyr[-1]))
        dx_total = 0.0; dy_total = 0.0
        for lvl in reversed(range(LEVELS+1)):
            p = prev_pyr[lvl]; c = cur_pyr[lvl]
            # apply Hanning window
            p_win = p * cv.resize(hw, (p.shape[1], p.shape[0]))
            c_win = c * cv.resize(hw, (c.shape[1], c.shape[0]))
            (dx, dy), _ = cv.phaseCorrelate(p_win, c_win)
            # accumulate (scale to next finer level)
            dx_total = (dx_total + dx) * 2.0 if lvl > 0 else dx_total + dx
            dy_total = (dy_total + dy) * 2.0 if lvl > 0 else dy_total + dy
        dx_pix, dy_pix = dx_total, dy_total
    else:
        p_win = prev_roi * hw
        c_win = cur_roi * hw
        (dx_pix, dy_pix), _ = cv.phaseCorrelate(p_win, c_win)

    # timing
    t = time.time(); dt = max(1e-3, t - last_t); last_t = t

    # pixel shift to planar velocity (meters/sec)
    vx = (dx_pix / dt) * (ALT_M / FX)
    vy = (dy_pix / dt) * (ALT_M / FY)

    # smooth
    vx_ema = (1 - SMOOTH_ALPHA) * vx_ema + SMOOTH_ALPHA * vx
    vy_ema = (1 - SMOOTH_ALPHA) * vy_ema + SMOOTH_ALPHA * vy

    # draw ROI and vector
    vis = frame.copy()
    cv.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 2)
    center = ((x0+x1)//2, (y0+y1)//2)
    tip = (int(center[0] + 30 * np.clip(dx_pix, -5, 5)),
           int(center[1] + 30 * np.clip(dy_pix, -5, 5)))
    cv.arrowedLine(vis, center, tip, (0,0,255), 2, tipLength=0.3)
    text = f"dx={dx_pix:.2f}px dy={dy_pix:.2f}px | vx={vx_ema:.2f} m/s vy={vy_ema:.2f} m/s"
    cv.putText(vis, text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
    cv.imshow("mouse-like VO (phase correlation)", vis)

    # prepare for next iter (update reference ROI every frame -> robust)
    prev_roi = cur_roi

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
