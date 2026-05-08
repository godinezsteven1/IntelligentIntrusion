from camera.camera_manager import CameraManager
from vision.detector import VisionDetector
import cv2


def main(): 
    camera = CameraManager()
    detector = VisionDetector()
    frame_count = 0 
    last_annotated_frame = None
    parsed_detections = []

    while True:
        frame = camera.read_frame()
    
        if frame is None:
            break  

        frame_count += 1
        
        if frame_count % 5 == 0: # infer every fifth frame
            parsed_detections, annotated_frame = detector.detect(frame)
            last_annotated_frame = annotated_frame
            print(parsed_detections)

            
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
