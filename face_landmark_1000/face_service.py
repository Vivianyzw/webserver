from face_landmark_1000.face_detector import *
from face_landmark_1000.face_landmark import *


class FaceLandmarkHandle(object):
    def __init__(self):
        self.face_detector_handle = FaceDetector('face_landmark_1000/model/FaceDetector.onnx')
        self.face_landmark_handle = FaceLandmark('face_landmark_1000/model/FaceLandmark.onnx')

    def run(self, image):
        detections, _ = self.face_detector_handle.run(image)
        if len(detections) == 0:
            return
        landmarks, states = self.face_landmark_handle.run(image, detections)
        if len(landmarks) > 0:
            result = self.face_landmark_handle.draw_result(image, landmarks)
            return result


class FaceDetectionHandle(object):
    def __init__(self):
        self.face_detector_handle = FaceDetector('face_landmark_1000/model/FaceDetector.onnx')

    def run(self, image):
        detections, _ = self.face_detector_handle.run(image)
        if len(detections) == 0:
            return
        result = self.face_detector_handle.draw_result(image, detections)
        return result


if __name__ == '__main__':
    handle = FaceDetectionHandle()
    image = cv2.imread(r'D:\Work\Github\webserver\static\images\1.jpeg')
    result = handle.run(image)
    cv2.imshow('', result)
    cv2.waitKey()