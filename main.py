#!/usr/bin/env python3

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load the image using OpenCV
# img_path = "/mnt/c/Users/Vincent/Downloads/Human_faces.jpg"
# img = cv2.imread(img_path)

# if img is None:
#     print(f"Error: Could not read image from {img_path}")
#     exit()

# height, width, _ = img.shape

# 2. Convert OpenCV BGR image to MediaPipe Image format
# mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# 3. FIX: Read model directly as binary data from disk into memory
model_filename = 'face_detector.tflite'
if not os.path.exists(model_filename):
    # Try alternative name if you haven't renamed the downloaded file
    model_filename = 'blaze_face_short_range.tflite'

if not os.path.exists(model_filename):
    print(f"Error: Neither 'face_detector.tflite' nor 'blaze_face_short_range.tflite' was found in {os.getcwd()}.")
    exit()

with open(model_filename, 'rb') as f:
    model_buffer = f.read()

webcam = cv2.VideoCapture(0) # webcam
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 4. Configure Face Detector using the memory buffer
base_options = python.BaseOptions(model_asset_buffer=model_buffer)
options = vision.FaceDetectorOptions(base_options=base_options)

# 5. Run detection
with vision.FaceDetector.create_from_options(options) as detector:
    while webcam.isOpened():
        ret, frame = webcam.read()
        if not ret:
            print("ignoring empty camera frame")
            continue

        cv2.imshow('blurred', frame)

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)

        # 6. Process results
        if detection_result.detections:
            for i, detection in enumerate(detection_result.detections):
                bbox = detection.bounding_box
                confidence = detection.categories[0].score
                
                # print(f"Face {i}: Confidence {confidence:.2f}")
                # print(f"Bounding Box: OriginX={bbox.origin_x}, OriginY={bbox.origin_y}, Width={bbox.width}, Height={bbox.height}")

                # 7. Face Anonymiser (Crop, Blur, Replace)
                x = max(0, bbox.origin_x)
                y = max(0, bbox.origin_y)
                w = min(bbox.width, width - x)
                h = min(bbox.height, height - y)

                face_roi = frame[y:y+h, x:x+w]

                if face_roi.size > 0:
                    # Apply strong blur
                    blurred_face = cv2.GaussianBlur(face_roi, (201, 201), 0)
                    frame[y:y+h, x:x+w] = blurred_face
                
            cv2.imshow('blurred', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # # 8. Save output image to your Windows Downloads folder
            # output_path = "/mnt/c/Users/Vincent/Downloads/face_anonymized.jpg"
            # cv2.imwrite(output_path, img)
            # print(f"\nSuccess! Anonymised image saved to: {output_path}")

    else:
        print("No faces detected.")

webcam.release()
cv2.destroyAllWindows()
