import cv2
import numpy as np
from PIL import Image

def blur_background(grayscale, img):    
    face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
    
    for (x_face, y_face, w_face, h_face) in face:
        roi_face= img[y_face: y_face+ h_face, x_face: x_face+ w_face]
        img = cv2.blur(img,(10,10))
        img[y_face: y_face+ h_face, x_face: x_face+ w_face]=roi_face      
    return img



def nothing(x):
    pass

def pix(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        r, g, b = rgbimg.getpixel((x,y))
        txt = str(r) + "," + str(g) + "," + str(b)
        print(txt)
        bg = np.zeros((200,400,3), np.uint8)
        bg[:,0:400] = (b,g,r)
        font = cv2.FONT_ITALIC
        cv2.putText(bg, txt, (10,100), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("rgb", bg)


print("###############################################")
print("Press:\n\t1.Blur background \n\t2.Colour tracking \n\t3.Colour Picker  \nPress space bar to exit")

while True:
    x = input("Enter: ")
    if x == "1":
        cap = cv2.VideoCapture(0) 
        cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        print("Press 'space bar' to quit")

        while True:
            ret, frame = cap.read()
            img = cv2.flip(frame,1)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            res = blur_background(grayscale, img)
            cv2.imshow('res', res) 
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break 
        cap.release() 
        cv2.destroyAllWindows()
    elif x == "2":
        cap = cv2.VideoCapture(0);
        cv2.namedWindow("Tracking")
        cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
        cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
        cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
        cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
        cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
        cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
        print("Press 'space bar' to quit")
        while True:
            _, frame = cap.read()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            l_h = cv2.getTrackbarPos("LH", "Tracking")
            l_s = cv2.getTrackbarPos("LS", "Tracking")
            l_v = cv2.getTrackbarPos("LV", "Tracking")

            u_h = cv2.getTrackbarPos("UH", "Tracking")
            u_s = cv2.getTrackbarPos("US", "Tracking")
            u_v = cv2.getTrackbarPos("UV", "Tracking")

            l_b = np.array([l_h, l_s, l_v])
            u_b = np.array([u_h, u_s, u_v])

            mask = cv2.inRange(hsv, l_b, u_b)

            res = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("res", res)
            cv2.imshow("mask", mask)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break 


        cap.release()
        cv2.destroyAllWindows()
    elif x == "3":
        print("1. Press 'space bar' to quit \n2.Left Mouse Click + 'c' to capture frame \n3.Double Click on captured image to see colour")
        cap = cv2.VideoCapture(0)
        while(True):
            ret, img = cap.read()
            frame = cv2.flip(img,1)
            cv2.imshow("feed",frame)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

            elif cv2.waitKey(1) & 0xFF == ord("c"):
                cv2.imwrite("1.png", frame)
                image = Image.open("1.png")
                rgbimg = image.convert("RGB")
                cv2.imshow("pic", frame)
                cv2.setMouseCallback("pic", pix)
            
            
         
        cap.release()
        cv2.destroyAllWindows()

    elif x == "5":
        break
    else:
        print("Please choose a valid option")
                

        
        
