

import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while(cap.isOpened()):

    ret_frame, frame = cap.read()

    frame = cv2.resize( frame, ( 896, 672))


    cv2.rectangle( frame, ( 50, 100), ( 350, 400), (0,255,0),0)
    hand_window = frame[ 100:400, 50:350]


    grey_hand_window = cv2.cvtColor(hand_window, cv2.COLOR_BGR2GRAY)


    blurred_grey_hand_window = cv2.GaussianBlur(grey_hand_window, (35, 35), 0)


    ret_thresh, thresh_by_otsu = cv2.threshold(blurred_grey_hand_window, 100, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh_by_otsu.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)


    hand_contour = max(contours, key = lambda x: cv2.contourArea(x))


    x, y, w, h = cv2.boundingRect(hand_contour)
    cv2.rectangle(hand_window, (x, y), (x+w, y+h), (0, 0, 255), 0)


    # finding convex hull
    hull = cv2.convexHull(hand_contour, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(hand_contour, hull)


    count_defects = 0


    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(hand_contour[s][0])
        end = tuple(hand_contour[e][0])
        far = tuple(hand_contour[f][0])

        
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        
        if angle <= 90:
            count_defects += 1
            cv2.circle(hand_window, far, 1, [0,0,255], -1)
        

        cv2.line(hand_window,start, end, [0,255,0], 2)



    if count_defects == 1:
        cv2.putText( frame,"I am Pranav", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, ( 0, 0, 255), 2)
    elif count_defects == 2:
        cv2.putText( frame, "I love food", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0, 255, 0), 2)
    elif count_defects == 3:
        cv2.putText( frame,"4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, ( 0, 255, 0), 2)
    elif count_defects == 4:
        cv2.putText( frame,"Hi", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, ( 0, 255, 0), 2)
    else:
        cv2.putText( frame,"Hello World", (50, 50),\
                    cv2.FONT_HERSHEY_SIMPLEX, 2, ( 0, 255, 0), 2)


    cv2.imshow('Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
