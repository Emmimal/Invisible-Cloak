import cv2
import numpy as np
import time

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Give the camera some time to warm up
time.sleep(3)

# Capture the background (without the cloak or object you want to make invisible)
for i in range(30):
    ret, background = cap.read()

# Flip the background image horizontally
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = np.flip(frame, axis=1)

    # Convert the frame to HSV color space for easier color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for the "cloak" (adjust the values for the cloak color)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Create masks to detect the color
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine the two masks
    mask = mask1 + mask2

    # Refine the mask (remove noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Create the inverse mask to segment out the cloak from the frame
    mask_inv = cv2.bitwise_not(mask)

    # Segment the cloak area
    cloak_area = cv2.bitwise_and(background, background, mask=mask)

    # Segment the non-cloak area
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine the two segments to create the final output
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

    # Show the output
    cv2.imshow("Invisibility Cloak", final_output)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
