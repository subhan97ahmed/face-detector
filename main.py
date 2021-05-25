import cv2 as cv

capture = cv.VideoCapture(0)

while True:
    istrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('video',gray)

    he_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_rec = he_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    print(f'no of faces = {len(face_rec)}')

    for (x, y, w, h) in face_rec:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        cv.putText(frame, "Face", (x, y), fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1.0, color=(0, 255, 0), thickness=2)

    cv.imshow('video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()