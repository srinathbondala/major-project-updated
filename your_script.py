from pyvirtualdisplay import Display

# Start a virtual display
display = Display(visible=0, size=(800, 600))
display.start()

import cv2

# Load an example image
img = cv2.imread("example.jpg")  # Replace with your image

if img is None:
    print("Error: Image not found!")
else:
    cv2.imshow("Test Window", img)  # OpenCV GUI
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()
