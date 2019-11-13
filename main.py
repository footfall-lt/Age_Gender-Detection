from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
import time
import cv2

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


def initialize_caffe_model():
    print('Loading models...')
    age_net = cv2.dnn.readNetFromCaffe("age_gender_models/deploy_age.prototxt", "age_gender_models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe("age_gender_models/deploy_gender.prototxt",
                                          "age_gender_models/gender_net.caffemodel")

    return age_net, gender_net


def capture_loop(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        print("Found " + str(len(faces)) + " face(s)")

    cv2.imshow("Image", image)

    key = cv2.waitKey(1) & 0xFF

    # clear the stream for the next frame
    rawCapture.truncate(0)

    if key == ord("q"):
        break


if __name__ == '__main__':
    age_net, gender_net = initialize_caffe_model()
