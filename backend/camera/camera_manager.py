# basic camera functionality and video campture 

import cv2


class CameraManager:

    """
    CameraManager owns hardware resources only. 
    """

    def __init__(
        self,
        source=0, # temp camera source for testing USB before CSI update 
        width=640,
        height=480,
        cam_name="Live Cam"):
            self.source = source
            self.width = width
            self.height = height
            self.cam_name = cam_name

            self.debug_open_failing = (
                f"backend/camera/camera_manager.py input source {self.source} camera cannot be opened or does not exist"
            )
            self.debug_frame_failing = (
                "backend/camera/camera_manager.py cannot receive frame (stream end?)"
            )

            self.cap = cv2.VideoCapture(self.source) # video stream input for now - usb

            if not self.cap.isOpened(): 
                #cam not open
                raise RuntimeError(self.debug_open_failing)
    
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)



    def read_frame(self):
        frame_success, frame = self.cap.read()

        if not frame_success:
            print(self.debug_frame_failing)
            return None
        return frame

        
    def release(self):
        self.cap.release()

# temp test loop 

if __name__ == "__main__":

    camera = CameraManager()

    while True:

        frame = camera.read_frame()

        if frame is None:
            break

        cv2.imshow(camera.cam_name, frame)

        if cv2.waitKey(1) == 27: # ESC key
            break 
    camera.release()
    cv2.destroyAllWindows()
