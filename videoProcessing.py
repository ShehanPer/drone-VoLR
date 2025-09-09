import cv2 as cv
import numpy as np

video_path = "recording_20250827_171942.mp4"

#Histogram Equalization
def hist_equalizer(frame):
  
        # Convert to YUV color space
        yuv_frame = cv.cvtColor(frame, cv.COLOR_BGR2YUV)

        # Equalize the histogram of the Y channel
        yuv_frame[:, :, 0] = cv.equalizeHist(yuv_frame[:, :, 0])

        # Convert back to BGR color space
        equalized_frame = cv.cvtColor(yuv_frame, cv.COLOR_YUV2BGR)

        # Display the original and equalized frames side by side
        combined_frame = np.hstack((frame, equalized_frame))
        cv.imshow('Original (Left) vs Equalized (Right)', combined_frame)

def intensity_trans(frame):
    
    t1 = np.linspace(0, 0, 150)

# Segment 2: input 60–179 → output 60–180
    t2 = np.linspace(100, 120, 30)

# Segment 3: input 180–255 → output 200–255
    t3 = np.linspace(200, 255,76 )  # 256 - (60+120) = 76

    t = np.concatenate((t1,t2,t3), axis=0).astype('uint8')

    # add gaussian smoothening
    frame = cv.GaussianBlur(frame, (5, 5), 0)

    transformed_frame = t[frame]
    combine_ = np.hstack((frame, transformed_frame))
    cv.imshow('Original (Left) vs Intensity Transformed (Right)', combine_)
     

cap = cv.VideoCapture(video_path)
while True:
    ret, frame = cap.read()

    if not ret: 
        print("Failed to read from video file.")
        break
    
    frame = cv.resize(frame, (320, 240))
    intensity_trans(frame)
    hist_equalizer(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
