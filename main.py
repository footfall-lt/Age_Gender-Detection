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
    age_net = cv2.dnn.readNetFromCaffe(
        "age_gender_models/deploy_age.prototxt",
        "age_gender_models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
        "age_gender_models/deploy_gender.prototxt",
        "age_gender_models/gender_net.caffemodel")

    return age_net, gender_net


if __name__ == '__main__':
    age_net, gender_net = initialize_caffe_model()
