import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    
    # Detect hands
    hands, img = detector.findHands(img, flipType=False)
    
    if hands:
        # First hand
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 landmarks (x, y, z)
        fingers1 = detector.fingersUp(hand1)
        handType1 = hand1["type"]  # "Left" or "Right"

        # Check for second hand
        if len(hands) == 2:
            # Second hand
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            fingers2 = detector.fingersUp(hand2)
            handType2 = hand2["type"]

            # Distance between index fingertips of two different hands
            length, info, img = detector.findDistance(
                lmList1[8][0:2],  # Index finger tip of first hand (x,y only)
                lmList2[8][0:2],  # Index finger tip of second hand (x,y only)
                img
            )

            print(f"{handType1} <-> {handType2} Distance: {length} pixels")
            print(f"{handType1} Fingers: {fingers1}")
            print(f"{handType2} Fingers: {fingers2}\n")
        
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()