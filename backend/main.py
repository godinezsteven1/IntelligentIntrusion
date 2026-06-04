from camera.camera_manager import CameraManager
from vision.detector import VisionDetector
from AI.face_engine import FaceEngine
import cv2


def main(): 
    camera = CameraManager(source="csi")
    detector = VisionDetector()
    face_engine = FaceEngine()
    frame_count = 0 
    last_annotated_frame = None

    while True:
        frame = camera.read_frame()
    
        if frame is None:
            break  

        frame_count += 1
        
        if frame_count % 5 == 0: # infer every fifth frame

            scene, annotated_frame = detector.detect(frame, frame_count)
            scene = face_engine.process(scene, frame)
            last_annotated_frame = annotated_frame
            
        if last_annotated_frame is not None:
            # display last annotated frame if exist
            cv2.imshow(camera.cam_name, last_annotated_frame)
        else:
            # fallback before first inference completes
            cv2.imshow(camera.cam_name, frame)


        if cv2.waitKey(1) == 27: #ESC
            break 

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
