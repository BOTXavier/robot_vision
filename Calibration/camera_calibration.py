import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
square_size = 20.0 #mm

objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2) * square_size
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


cv.namedWindow("img", cv.WINDOW_NORMAL)

extensions = ['*.jpg', '*.jpeg', '*.png']
images = [f for ext in extensions for f in glob.glob(ext)]


for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,9), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

       # <-- reste ouvert
cv.destroyAllWindows()


#Now that we have our object points and image points, we are ready to go for calibration. We can use the function, 
# cv.calibrateCamera() which returns the camera matrix, distortion coefficients, rotation and translation vectors etc. 
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Matrice : ")
print (mtx)
print("Distorsion vector")
print(dist)
