import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width (correct property ID: 3 for width)
cap.set(4, 720)   # Height (correct property ID: 4 for height)

detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    
    # Detect hands (single detection call)
    hands, img = detector.findHands(img, flipType=False)
    
    if hands:
        # Get first hand
        hand1 = hands[0]
        
        # Correct dictionary keys
        lmList1 = hand1["lmList"]         # Correct key name
        bbox = hand1["bbox"]              # Bounding box coordinates
        centerPoint1 = hand1["center"]    # Center point of hand
        handType1 = hand1["type"]         # Hand type ("Left" or "Right")

        # Each lmList contains 21 landmarks (x, y, z coordinates)
        print(f"Landmarks count: {len(lmList1)}, First landmark: {lmList1[0]}")
        print(bbox) #shows the 4 value
        print(centerPoint1) #shows the center value
        print(type) #shows which hand is there left or right
        
    cv2.imshow("Image", img)
    cv2.waitKey(1)