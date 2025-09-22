import cv2
import mediapipe as mp
import time

# Initialize the video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize the hands and drawing utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialize the previous time and current time
pTime = 0
cTime = 0

# Main loop for capturing frames
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # If there are any hand landmarks, draw them
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                
                # Get the height, width, and channel of the image
                height, width, channel = img.shape
                centerX, centerY = int(lm.x * width), int(lm.y * height)
                print(id, centerX, centerY)

                # If the id is 0 (wrists), draw a circle on the image
                if id ==0:
                    cv2.circle(img, (centerX, centerY), 15, (255, 0, 255), cv2.FILLED)
    
            # Draw the hand landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate the frames per second
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Put the frames per second on the image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Show the image
    cv2.imshow("Image", img)

    # If the user presses the 'q' key, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()