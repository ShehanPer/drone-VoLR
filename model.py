import torch 
import torch.nn as nn
import cv2 as cv
import numpy as np

class Letter_recognition(nn.Module):
    def __init__(self):
        super(Letter_recognition, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        dummy_input = torch.randn(1, 3, 64, 64)
        output_size = self.features(dummy_input).flatten(1).shape[1]
        self.fully_connected = nn.Sequential(
            nn.Linear(output_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 9)
        )
    def forward(self, x):
        x = self.features(x)
        X = torch.flatten(x, 1)
        X = self.fully_connected(X)
        return X
# --- End model class ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

model = torch.load("letter_cnn_best.pt", map_location=device)
model.to(device)
model.eval()

#Helper functions
def gamma_correction(image):
    gamma = 0.2
    image_lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    G = np.array([((i / 255) ** (1 / gamma)) * 255 for i in range(256)]).astype("uint8")
    image_lab[:, :, 0] = cv.LUT(image_lab[:, :, 0], G)
    image_corrected = cv.cvtColor(image_lab, cv.COLOR_LAB2BGR)
    return image_corrected

def average_filter(image):
    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    smoothed_image = cv.filter2D(image, -1, kernel)
    return smoothed_image

def preprocess_image(img_bgr):
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_rgb = cv.resize(img_rgb, (64, 64))
    img_rgb = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img_rgb = (img_rgb - mean) / std
    img_rgb = np.transpose(img_rgb, (2, 0, 1))
    img_tensor = torch.tensor(img_rgb, dtype=torch.float32).unsqueeze(0)  # <-- Ensure float32
    return img_tensor

# get video 
video_dir = "recording_20250827_171942.mp4"
cap = cv.VideoCapture(video_dir)

# Get video properties
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# Define codec and create VideoWriter
output_path = "letter_recognition_output.mp4"  # or .mp4
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # You can also try 'mp4v'
out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


class_names = ["A","B","C","D","E","F","G","H","I"]

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error.")
        break

    H, W = frame.shape[:2]
    print(W)
    frame_center_x, frame_center_y = W//2, H//2

    gamma_image = gamma_correction(frame)
    filtered_image = average_filter(gamma_image)
    img_gray = cv.cvtColor(filtered_image, cv.COLOR_BGR2GRAY)

    _, binary = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    image_with_contours = gamma_image.copy()
    for contour in contours:
        if cv.contourArea(contour) > 1500:
            x, y, w, h = cv.boundingRect(contour)
            if 0.6<cv.contourArea(contour)/(w*h)<1.0:
                cv.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
                box_center_x, box_center_y= x+w//2, y+h//2

                distance = np.sqrt((frame_center_x-box_center_x)**2 + (frame_center_y-box_center_y)**2 )
                distance = round(distance, 2)

                # feed cropped regions to model
                letter_img = gamma_image[y:y+h, x:x+w]
                letter_img = preprocess_image(letter_img).to(device)
                with torch.no_grad():
                    outputs = model(letter_img)
                    predicted = outputs.argmax(dim=1)
                    predicted_letter = class_names[predicted.item()]
                cv.putText(image_with_contours, predicted_letter, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv.putText(image_with_contours, str(distance), (x+15, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
                cv.line(image_with_contours,(box_center_x,box_center_y),(frame_center_x,frame_center_y),color=(0, 0, 255), thickness=2)

    cv.imshow("Letter Recognition", image_with_contours)
    out.write(image_with_contours)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release() 
cv.destroyAllWindows()
print("Processing complete. Output saved to", output_path)