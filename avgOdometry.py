import numpy as np
import cv2 as cv


# Parameters
USE_PYRAMID = True               # coarse-to-fine for bigger shifts
LEVELS = 2
ROI_W, ROI_H = 80, 80          # small box size for compute phase correlation

# Helper Functions
def hanningWindow(shape):
    return cv.createHanningWindow((shape[1], shape[0]), cv.CV_32F)

def rois(img, w, h):
    H, W = img.shape[:2]

    roi_1 = [5, 5, w + 5, h + 5]
    roi_2 = [W - w - 5, 5, W - 5, h + 5]
    roi_3 = [5, H - h - 5, w + 5, H - 5]
    roi_4 = [W - w - 5, H - h - 5, W - 5, H - 5]
    roi_5 = [W // 2 - w // 2, H // 2 - h // 2, W // 2 + w // 2, H // 2 + h // 2]

    return [roi_1, roi_2, roi_3, roi_4, roi_5]

def calcDisplacement(prev_roi, cur_roi, hw):
    if USE_PYRAMID:
        prev_pyr = [prev_roi]; cur_pyr = [cur_roi]
        for _ in range(LEVELS):
            prev_pyr.append(cv.pyrDown(prev_pyr[-1]))
            cur_pyr.append(cv.pyrDown(cur_pyr[-1]))
        dx_total = 0.0; dy_total = 0.0

        for lvl in reversed(range(LEVELS+1)):
            p = prev_pyr[lvl]; c = cur_pyr[lvl]

            p_win = p*cv.resize(hw, (p.shape[1], p.shape[0]))
            c_win = c*cv.resize(hw, (c.shape[1], c.shape[0]))

            (dx, dy),_ = cv.phaseCorrelate(p_win, c_win)

            dx_total = (dx_total + dx) * 2.0 if lvl > 0 else dx_total + dx
            dy_total = (dy_total + dy) * 2.0 if lvl > 0 else dy_total + dy
        
        dx_pix, dy_pix = dx_total, dy_total
    else:
        p_win = prev_roi * hw
        c_win = cur_roi * hw
        (dx_pix, dy_pix) = cv.phaseCorrelate(p_win, c_win)

    return dx_pix, dy_pix

url = "http://192.168.8.102:8080/video"
cap = cv.VideoCapture(url) ##################### or use camera index or video file or have to change this when using RASPI camera 

ret, frame = cap.read()
if not ret:
    print("Failed to read from camera or video file.")
    cap.release()
    exit()
frame = cv.resize(frame, (320, 240)) 
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
frame_gray = cv.GaussianBlur(frame_gray, (5, 5), 0) 
prev_gray = frame_gray.copy()

hw = hanningWindow((ROI_H, ROI_W))

while True:

    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera or video file.")
        cap.release()
        exit()

    frame = cv.resize(frame, (320, 240)) 
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, (5, 5), 0) # used small gaussian filter to reduce the noise effects

    rois_list = rois(frame_gray, ROI_W, ROI_H)
    cur_gray = frame_gray.copy()

    displacements = [] # temporary list to store displacements from all ROIs
    for roi in rois_list:
        x0, y0, x1, y1 = roi
        prev_roi = prev_gray[y0:y1, x0:x1].astype(np.float32)
        cur_roi = cur_gray[y0:y1, x0:x1].astype(np.float32)
        

        dx, dy = calcDisplacement(prev_roi, cur_roi, hw)  # Initial call to set up the previous ROI
        displacements.append([dx, dy])

        # Per-ROI arrows
        dx_scaled = np.clip(dx * 5, -15, 15)
        dy_scaled = np.clip(dy * 5, -15, 15)
        center = (x0 + ROI_W // 2, y0 + ROI_H // 2)
        tip = (int(center[0] + dx_scaled), int(center[1] + dy_scaled))
        cv.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv.arrowedLine(frame, center, tip, (255, 0, 0), 1, tipLength=0.3)

    
    mean_dx = np.median([d[0] for d in displacements])
    mean_dy = np.median([d[1] for d in displacements])

    print(f"Mean Displacement: dx = {mean_dx:.2f}, dy = {mean_dy:.2f}")
    # plot on frame 
    scale = 10
    max_len = 40
    dx_scaled = np.clip(mean_dx * scale, -max_len, max_len)
    dy_scaled = np.clip(mean_dy * scale, -max_len, max_len)
    center = (160, 120)
    tip = (int(center[0] + dx_scaled), int(center[1] + dy_scaled))
    cv.arrowedLine(frame, center, tip, (0, 0, 255), 2, tipLength=0.3)

    cv.putText(frame, f"dx={mean_dx:.2f}, dy={mean_dy:.2f}",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.imshow("Multi-ROI Phase Correlation", frame)
    prev_gray = cur_gray.copy()


    k = cv.waitKey(1) & 0xFF
    if k == 27:  # Press 'ESC' to exit
        break

cap.release()
cv.destroyAllWindows()