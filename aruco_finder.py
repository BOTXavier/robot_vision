#!/usr/bin/env python3
import cv2
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..')) # Avoids ModuleNotFoundError when finding generated folder
# import ecal.nanobind_core as ecal_core
# from ecal.msg.proto.core import Subscriber as ProtoSubscriber
# from ecal.msg.proto.core import Publisher as ProtoPublisher
# from ecal.msg.common.core import ReceiveCallbackData
# from generated.robot_state_pb2 import Position_aruco
# from generated import CompressedImage_pb2 as cipb
# from google.protobuf.timestamp_pb2 import Timestamp
import argparse
from enum import Enum
from scipy.spatial.transform import Rotation
import time
from threading import Event


class Source(Enum):
    CAM = 0
    VIDEO = 1
    ECAL = 2


class ArucoFinder:
    def __init__(self, name, src_type, src, arucos, display):
        # if not ecal_core.is_initialized():
        #     ecal_core.initialize("arucoFinder")
        
        self.name = name
        self.src_type = src_type
        self.src = src
        self.arucos = arucos
        self.display = display
        
        self.event = Event()
        self.img = None     # img received from eCAL

        # self.aruco_pub = ProtoPublisher(Position_aruco, "Arucos")
        # if self.display:
        #     self.cam_pub = ProtoPublisher(cipb.CompressedImage, "images_"+str(self.name))
        
        # ArUco settings (API OpenCV 4.7+)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.open_capture()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    #     if src_type == Source.ECAL:
    #         self.sub.remove_receive_callback()
    
    def open_capture(self):
        if src_type == Source.CAM:
            self.cap = cv2.VideoCapture(src)
            if self.cap is None:
                print(f"Failed to open {self.name} with id {src}")
                exit(-1)
            if args.width is not None and args.height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            if args.fps is not None:
                print(f"setting fps at {args.fps}...")
                self.cap.set(cv2.CAP_PROP_FPS, args.fps)
            w, h, f = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Opened camera with resolution {w}x{h} at {f}fps!\n")
        elif src_type == Source.VIDEO:
            self.cap = cv2.VideoCapture(src)
            if self.cap is None:
                print(f"Failed to open {self.name} with id {src}")
                return
            w, h = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Opened video with resolution {w}x{h}!\n")
        # elif src_type == Source.ECAL:
        #     self.sub = ProtoSubscriber(cipb.CompressedImage, src)
        #     self.sub.set_receive_callback(self.on_img)

    def getCalibration(self, w, h):
        """Provide Calibration Matrix and distance coefs as .npy file"""
        # Charger la calibration
        f_mat = f'./Calibration/{self.name}_matrix_{w}x{h}.npy'
        f_coef = f'./Calibration/{self.name}_coeffs_{w}x{h}.npy'
        self.camera_matrix = np.load(f_mat)
        self.dist_coeffs = np.load(f_coef)
    
    # def on_img(self, pub_id: ecal_core.TopicId, data: ReceiveCallbackData[cipb.CompressedImage]):
    #     nparr = np.frombuffer(data.message.data, np.uint8)
    #     self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #     self.event.set()

    # def send_processed_frame(self, frame):
    #     img_encode = cv2.imencode(".jpg", frame)[1]
    #     byte_encode = img_encode.tobytes()
    #     ci = cipb.CompressedImage(timestamp=Timestamp(), data=byte_encode, format='jpeg')
    #     self.cam_pub.send(ci)

    def estimatePoseSingleMarkers(self,corners, marker_size, mtx, distortion):
        '''
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        '''
        marker_points = np.array([[600, 600, 0],
                                [2400, 600, 0],
                                [2400, 1400, 0],
                                [600, 1400, 0]], dtype=np.float32)
        trash = []
        rvecs = []
        tvecs = []
        
        for c in corners:
            ok, rvec, tvec = cv2.solvePnP(
                marker_points,
                c,
                mtx,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            trash.append(ok)

            # --- reshape to match OpenCV format ---
            rvec = rvec.reshape(1, 3)
            tvec = tvec.reshape(1, 3)

            rvecs.append(rvec)
            tvecs.append(tvec)

        # Convert lists to arrays: final shapes (N,1,3)
        rvecs = np.array(rvecs, dtype=np.float32)
        tvecs = np.array(tvecs, dtype=np.float32)

        return rvecs, tvecs, trash
    
    def estimatePoseFromCenters(self, centers, mtx, distortion):
        """
        Estimate camera pose from 4 points (centers of 4 ArUco) and their positions in world coordinates.

        Args:
            centers (list of np.ndarray): liste des centres détectés dans l'image, shape (4,2)
            world_positions (list of np.ndarray): liste des positions connues des marqueurs, shape (4,3)
            mtx (np.ndarray): matrice intrinsèque caméra
            distortion (np.ndarray): coefficients de distorsion caméra

        Returns:
            rvec (np.ndarray): vecteur rotation (3x1)
            tvec (np.ndarray): vecteur translation (3x1)
            ok (bool): True si solvePnP a réussi
        """
        object_points = np.array([
                                    [600, 1400, 0],   # marqueur 20
                                    [2400, 1400, 0],  # marqueur 21
                                    [600, 600, 0],    # marqueur 22
                                    [2400, 600, 0]    # marqueur 23
                                ], dtype=np.float32)
        if len(centers) < 4:
            print("Pas assez de points pour solvePnP")
            return None, None, False

        # Convertir en np.array float32
        image_points = np.array(centers, dtype=np.float32)

        rvecs = []
        tvecs = []

        # SolvePnP
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            mtx,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if ok:
            rvec = rvec.reshape(1, 3)
            tvec = tvec.reshape(1, 3)
        

        rvecs.append(rvec)
        tvecs.append(tvec)

        return rvecs, tvecs, ok
    

    def camera_pose_in_world(self, rvec, tvec, marker_pos_world):
        """
        Calcule la position et l'orientation de la caméra dans le repère monde.
        
        Args:
            rvec (np.ndarray): vecteur rotation (3x1) du marqueur dans le repère caméra.
            tvec (np.ndarray): vecteur translation (3x1) du marqueur dans le repère caméra.
            marker_pos_world (np.ndarray): position du centre du marqueur dans le repère monde (3x1).

        Returns:
            cam_pos_world (np.ndarray): position de la caméra dans le repère monde (3x1).
            cam_rot_world (np.ndarray): rotation de la caméra dans le repère monde (3x3).
        """
        # Convertir rvec en matrice de rotation
        R_marker_cam, _ = cv2.Rodrigues(rvec)  # marker -> camera
        
        # Position de la caméra dans le repère monde
        cam_pos_world = marker_pos_world.reshape(3,1) - R_marker_cam.T @ tvec.reshape(3,1)
        
        # Orientation de la caméra dans le repère monde
        # R_marker_cam: marker -> camera
        # R_cam_marker = R_marker_cam.T
        # Pour passer à world: R_cam_world = R_cam_marker @ R_world_marker
        # Si le marqueur est aligné avec le monde, R_world_marker = I
        cam_rot_world = R_marker_cam.T  # si pas de rotation du marqueur dans le monde

        print("pose : ")
        print(cam_pos_world)
        print("rot : " )
        print(cam_rot_world)

        return cam_pos_world, cam_rot_world
    
    def camera_pose_in_world_from_tags(self, rvec, tvec):
        """
        Calcule la position et l'orientation de la caméra dans le repère monde.
        
        Args:
            rvec (np.ndarray): vecteur rotation (3x1) du marqueur dans le repère caméra.
            tvec (np.ndarray): vecteur translation (3x1) du marqueur dans le repère caméra.
            marker_pos_world (np.ndarray): position du centre du marqueur dans le repère monde (3x1).

        Returns:
            cam_pos_world (np.ndarray): position de la caméra dans le repère monde (3x1).
            cam_rot_world (np.ndarray): rotation de la caméra dans le repère monde (3x3).
        """
        # Convertir rvec en matrice de rotation
        R_marker_cam, _ = cv2.Rodrigues(rvec)  # marker -> camera
        
        # Position de la caméra dans le repère monde
        cam_pos_world =  - R_marker_cam.T @ tvec.reshape(3,1)
        
        # Orientation de la caméra dans le repère monde
        # R_marker_cam: marker -> camera
        # R_cam_marker = R_marker_cam.T
        # Pour passer à world: R_cam_world = R_cam_marker @ R_world_marker
        # Si le marqueur est aligné avec le monde, R_world_marker = I
        cam_rot_world = R_marker_cam.T  # si pas de rotation du marqueur dans le monde

        print("pose : ")
        print(cam_pos_world)
        print("rot : " )
        print(cam_rot_world)

        return cam_pos_world, cam_rot_world
    

    def process(self, frame):
        """Call it in a while true loop"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection ArUco
        detected_corners, detected_ids, rejected = self.aruco_detector.detectMarkers(gray)

        if self.display:
            cv2.aruco.drawDetectedMarkers(frame, detected_corners, detected_ids)
        
        cv2.aruco.drawDetectedMarkers(frame, detected_corners, detected_ids)

        if detected_corners:
            xs, ys, zs, aruIds= [],[],[],[]
            qws, qxs, qys, qzs = [],[],[],[]
            centers = []
            centers_id = []
            for corners, id in zip(detected_corners, detected_ids):
                id = id[0]
                print(id)
                if id not in self.arucos:
                    continue
                center = corners[0].mean(axis=0)
                centers.append(center)
                centers_id.append(id)

                size = self.arucos[id]
            
            combined = list(zip(centers_id, centers))

            # Sort by id
            combined.sort(key=lambda x: x[0])
            centers_id, centers = zip(*combined)

            centers_id = list(centers_id)
            centers = list(centers)

            if centers :        
                rvecs, tvecs, _ = self.estimatePoseFromCenters(centers, self.camera_matrix, self.dist_coeffs)
                #rvecs, tvecs, _ = self.estimatePoseSingleMarkers(corners, size, self.camera_matrix, self.dist_coeffs)
 
                if tvecs is not None:
                    rv, tv = rvecs[0], tvecs[0]
                    
                    self.camera_pose_in_world_from_tags(rv, tv)

                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rv, tv, size)
                    xs.append(tv[0][0])
                    ys.append(tv[0][1])
                    zs.append(tv[0][2])
                    aruIds.append(id)

                    # Convert rvec to rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(np.array(rv))
                    r =  Rotation.from_matrix(rotation_matrix)
                    (qx, qy, qz, qw) = r.as_quat()
                    qxs.append(qx)
                    qys.append(qy)
                    qzs.append(qz)
                    qws.append(qw)
            #self.arucoFound = Position_aruco(x=xs, y=ys, z=zs, qx=qxs, qy=qys, qz=qzs, qw=qws, ArucoId=aruIds, cameraName=self.name)
            #self.aruco_pub.send(self.arucoFound)
        return frame
    

    def run(self):
        while True:
            if self.src_type == Source.CAM or self.src_type == Source.VIDEO:
                ret, frame = self.cap.read()
            else:
                self.event.wait()
                frame = self.img.copy()
            
            if self.camera_matrix is None:
                h, w, _ = frame.shape
                self.getCalibration(w, h)
            
            processed = self.process(frame)
            if self.display:
                self.send_processed_frame(processed)
            
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow(f"ArucoFinder - {self.name}", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='camera name')
    parser.add_argument('-c', '--cam', type=int, help='Camera ID', default=None)
    parser.add_argument('-v', '--video', help='Video file', default=None)
    parser.add_argument('-t', '--topic', help='eCAL topic', default=None)
    parser.add_argument('-d', '--display', action='store_true', default=False, help='send annotated images over ecal')
    parser.add_argument('-W', '--width', type=int, help='image width', default=None)
    parser.add_argument('-H', '--height', type=int, help='image height', default=None)
    parser.add_argument('-f', '--fps', type=int, help='framerate', default=None)
    args = parser.parse_args()


    if args.cam is not None:
        src_type = Source.CAM
        src = args.cam
    elif args.video is not None:
        src_type = Source.VIDEO
        src = args.video

    else:
        print("Please specify the source: cam, video or ecal topic.")
    
    arucos = {20:100, 21:100, 22:100, 23:100}
    known_markers = {
        20: np.array([600, 1400, 0]),   # x=600, y=1400, z=0
        21: np.array([2400, 1400, 0]),  # x=2400, y=1400, z=0
        22: np.array([600, 600, 0]),    # x=600, y=600, z=0
        23: np.array([2400, 600, 0])    # x=2400, y=600, z=0
    }


    


    with ArucoFinder(args.name, src_type, src, arucos, args.display) as af:
        af.run()#k