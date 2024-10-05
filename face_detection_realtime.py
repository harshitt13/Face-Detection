import cv2
import time
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Buffer for smoothing
landmark_buffer = []
bbox_buffer = []
buffer_size = 5

def smooth_coordinates(buffer, new_coords):
    buffer.append(new_coords)
    if len(buffer) > buffer_size:
        buffer.pop(0)
    return np.mean(buffer, axis=0)

def process_frame(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            x_max = int((bboxC.xmin + bboxC.width) * w)
            y_max = int((bboxC.ymin + bboxC.height) * h)
                    
            x_min = max(0, x_min - 20)
            y_min = max(0, y_min - 20)
            x_max = min(w, x_max + 20)
            y_max = min(h, y_max + 20)
            
            bbox_coords = [x_min, y_min, x_max, y_max]
            smoothed_bbox = smooth_coordinates(bbox_buffer, bbox_coords)
            x_min, y_min, x_max, y_max = map(int, smoothed_bbox)
            
            face_img = img[y_min:y_max, x_min:x_max]
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_mesh_results = face_mesh.process(face_img_rgb)
            
            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                    smoothed_landmarks = smooth_coordinates(landmark_buffer, landmarks)
                    
                    for idx, landmark in enumerate(smoothed_landmarks):
                        face_landmarks.landmark[idx].x = landmark[0]
                        face_landmarks.landmark[idx].y = landmark[1]
                        face_landmarks.landmark[idx].z = landmark[2]
                    
                    if face_landmarks and len(face_landmarks.landmark) > max(max(connection) for connection in mp_face_mesh.FACEMESH_TESSELATION):
                        mp_drawing.draw_landmarks(
                            image=face_img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    if face_landmarks and len(face_landmarks.landmark) > max(max(connection) for connection in mp_face_mesh.FACEMESH_CONTOURS):
                        mp_drawing.draw_landmarks(
                            image=face_img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    if face_landmarks and len(face_landmarks.landmark) > max(max(connection) for connection in mp_face_mesh.FACEMESH_IRISES):
                        mp_drawing.draw_landmarks(
                            image=face_img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
    
    return img

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ptime = 0

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        img = cv2.flip(img, 1)  # Mirror the image
        img = process_frame(img)
        
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        cv2.imshow('Face Detection', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
