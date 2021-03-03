import cv2

def blur_background(grayscale, img):    
    face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
    
    for (x_face, y_face, w_face, h_face) in face:
        roi_face= img[y_face: y_face+ h_face, x_face: x_face+ w_face]
        img = cv2.blur(img,(10,10))
        img[y_face: y_face+ h_face, x_face: x_face+ w_face]=roi_face      
    return img


cap = cv2.VideoCapture(0) 
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Press 'space bar' to quit")

while True:
    ret, frame = cap.read()
    img = cv2.flip(frame,1)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    final = blur_background(grayscale, img)
    cv2.imshow('Video', final) 
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break 
cap.release() 
cv2.destroyAllWindows() 







