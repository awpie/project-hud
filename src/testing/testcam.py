import cv2

cap = cv2.VideoCapture(1)  # Device 1 = DroidCam?

if not cap.isOpened():
    print("Failed to open video device")
    exit()

while True:
    ret, frame = cap.read()
    #print("Frame shape:", frame.shape)
    #print("Frame dtype:", frame.dtype)
    #print("Frame mean pixel value:", frame.mean())
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow('DroidCam Feed', frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()