#!/usr/bin/env python3
"""
Camera Calibration for ArUco Marker Detection
Works with OpenCV 4.X
"""

import numpy as np
import cv2
import glob
import os
import json


class CameraCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
        """
        Initialize the camera calibrator.

        Args:
            checkerboard_size: Tuple of (columns, rows) of inner corners in checkerboard
            square_size: Size of checkerboard square in meters (default 25mm)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size

        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... etc
        self.objp = np.zeros(
            (checkerboard_size[0] * checkerboard_size[1], 3), np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0 : checkerboard_size[0], 0 : checkerboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= square_size

        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane

    def capture_images(self, num_images=20, camera_id=0):
        """
        Capture calibration images from webcam.

        Args:
            num_images: Number of calibration images to capture
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return False

        # Set camera to 1920x1080 resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Verify actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")

        print(f"\nCalibration Image Capture")
        print(f"========================")
        print(
            f"Position the checkerboard ({self.checkerboard_size[0]}x{self.checkerboard_size[1]}) in front of the camera"
        )
        print(f"Press SPACE to capture image ({num_images} needed)")
        print(f"Press 'q' to quit")
        print(f"\nTips:")
        print(f"- Capture images from different angles and distances")
        print(f"- Cover different areas of the camera view")
        print(f"- Ensure the checkerboard is well-lit and in focus\n")

        captured = 0
        os.makedirs("calibration_images", exist_ok=True)

        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find the checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(
                gray, self.checkerboard_size, None
            )

            display_frame = frame.copy()

            # Draw corners if found
            if ret_corners:
                cv2.drawChessboardCorners(
                    display_frame, self.checkerboard_size, corners, ret_corners
                )
                cv2.putText(
                    display_frame,
                    "Checkerboard detected! Press SPACE to capture",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display_frame,
                    "Move checkerboard into view",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                display_frame,
                f"Captured: {captured}/{num_images}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Camera Calibration - Capture Images", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and ret_corners:
                # Save the image
                filename = f"calibration_images/calib_{captured:02d}.jpg"
                cv2.imwrite(filename, frame)
                captured += 1
                print(f"✓ Captured image {captured}/{num_images}")

            elif key == ord("q"):
                print("\nCalibration cancelled by user")
                break

        cap.release()
        cv2.destroyAllWindows()

        return captured >= num_images

    def calibrate_from_images(self, image_path="calibration_images/*.jpg"):
        """
        Perform calibration using captured images.

        Args:
            image_path: Path pattern to calibration images

        Returns:
            Tuple of (camera_matrix, distortion_coefficients, rvecs, tvecs)
        """
        images = glob.glob(image_path)

        if len(images) == 0:
            print(f"Error: No images found at {image_path}")
            return None

        print(f"\nProcessing {len(images)} calibration images...")

        img_shape = None
        successful_images = 0

        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img_shape is None:
                img_shape = gray.shape[::-1]

            # Find the checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

            if ret:
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria
                )

                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_refined)
                successful_images += 1
                print(f"✓ Processed: {os.path.basename(fname)}")
            else:
                print(f"✗ Failed to find corners: {os.path.basename(fname)}")

        if successful_images < 3:
            print(
                f"\nError: Need at least 3 successful images, got {successful_images}"
            )
            return None

        print(f"\nCalibrating camera with {successful_images} images...")

        # Ensure arrays are in correct format for OpenCV 4.X
        objpoints_array = [np.asarray(pts, dtype=np.float32) for pts in self.objpoints]
        imgpoints_array = [np.asarray(pts, dtype=np.float32) for pts in self.imgpoints]

        # Perform camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints_array,
            imgpoints_array,
            img_shape,
            None,  # cameraMatrix
            None,  # distCoeffs
        )

        if not ret:
            print("Error: Calibration failed")
            return None

        print("✓ Calibration successful!")

        # Calculate reprojection error
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(
                imgpoints2
            )
            mean_error += error

        mean_error /= len(self.objpoints)
        print(f"Mean reprojection error: {mean_error:.4f} pixels")

        return camera_matrix, dist_coeffs, rvecs, tvecs

    def save_calibration(
        self, camera_matrix, dist_coeffs, filename="camera_calibration.json"
    ):
        """
        Save calibration data to a JSON file.

        Args:
            camera_matrix: Camera matrix
            dist_coeffs: Distortion coefficients
            filename: Output filename
        """
        calibration_data = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "image_width": int(camera_matrix[0, 2] * 2),
            "image_height": int(camera_matrix[1, 2] * 2),
        }

        with open(filename, "w") as f:
            json.dump(calibration_data, f, indent=4)

        print(f"\n✓ Calibration data saved to {filename}")

        # Also save as numpy file for easier loading
        np.savez(
            filename.replace(".json", ".npz"),
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        print(f"✓ Calibration data also saved to {filename.replace('.json', '.npz')}")

    def load_calibration(self, filename="camera_calibration.npz"):
        """
        Load calibration data from a file.

        Args:
            filename: Calibration file path

        Returns:
            Tuple of (camera_matrix, distortion_coefficients)
        """
        if filename.endswith(".npz"):
            data = np.load(filename)
            return data["camera_matrix"], data["dist_coeffs"]
        elif filename.endswith(".json"):
            with open(filename, "r") as f:
                data = json.load(f)
            camera_matrix = np.array(data["camera_matrix"])
            dist_coeffs = np.array(data["distortion_coefficients"])
            return camera_matrix, dist_coeffs
        else:
            raise ValueError("Unsupported file format. Use .npz or .json")


def print_calibration_info(camera_matrix, dist_coeffs):
    """Print calibration information in a readable format."""
    print("\n" + "=" * 60)
    print("CAMERA CALIBRATION RESULTS")
    print("=" * 60)
    print("\nCamera Matrix (Intrinsic Parameters):")
    print(camera_matrix)
    print(
        f"\nFocal Length (fx, fy): ({camera_matrix[0, 0]:.2f}, {camera_matrix[1, 1]:.2f})"
    )
    print(
        f"Principal Point (cx, cy): ({camera_matrix[0, 2]:.2f}, {camera_matrix[1, 2]:.2f})"
    )
    print(f"\nDistortion Coefficients:")
    print(f"k1, k2, p1, p2, k3 = {dist_coeffs.ravel()}")
    print("=" * 60 + "\n")


def test_calibration(camera_matrix, dist_coeffs, camera_id=0):
    """
    Test the calibration by showing undistorted video feed.

    Args:
        camera_matrix: Camera matrix
        dist_coeffs: Distortion coefficients
        camera_id: Camera device ID
    """
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera to native resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Verify actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nCamera resolution: {actual_width}x{actual_height}")

    print("\nTesting calibration...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        # Undistort the image
        undistorted = cv2.undistort(
            frame, camera_matrix, dist_coeffs, None, new_camera_matrix
        )

        # Crop the image based on ROI
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            undistorted = undistorted[y : y + h_roi, x : x + w_roi]

        # Calculate proper display size maintaining aspect ratio
        display_width = 960  # Half of 1920
        display_height = 540  # Half of 1080

        # Resize both frames to same display size
        frame_display = cv2.resize(frame, (display_width, display_height))
        undistorted_display = cv2.resize(undistorted, (display_width, display_height))

        # Create side-by-side comparison
        combined = np.hstack([frame_display, undistorted_display])

        # Add labels
        cv2.putText(
            combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            combined,
            "Undistorted",
            (display_width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Calibration Test - Original vs Undistorted", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main function to run camera calibration."""
    print("\n" + "=" * 60)
    print("CAMERA CALIBRATION FOR ARUCO MARKER DETECTION")
    print("OpenCV 4.X Compatible")
    print("=" * 60)

    # Initialize calibrator
    # Standard checkerboard: 9x6 inner corners, 25mm squares
    calibrator = CameraCalibrator(checkerboard_size=(9, 6), square_size=0.025)

    print("\nYou'll need a checkerboard calibration pattern.")
    print("You can download one from:")
    print("https://github.com/opencv/opencv/blob/4.x/doc/pattern.png")
    print('\nPrint it on US Letter paper (8.5" x 11")')
    print("The pattern should fit well on letter-sized paper.")

    input("\nPress Enter when ready to start...")

    # Step 1: Capture calibration images
    print("\n--- Step 1: Capture Calibration Images ---")
    success = calibrator.capture_images(num_images=20, camera_id=0)

    if not success:
        print("\nCalibration aborted. Not enough images captured.")
        return

    # Step 2: Perform calibration
    print("\n--- Step 2: Perform Calibration ---")
    result = calibrator.calibrate_from_images()

    if result is None:
        print("\nCalibration failed.")
        return

    camera_matrix, dist_coeffs, rvecs, tvecs = result

    # Step 3: Save calibration
    print("\n--- Step 3: Save Calibration Data ---")
    calibrator.save_calibration(camera_matrix, dist_coeffs)

    # Print calibration info
    print_calibration_info(camera_matrix, dist_coeffs)

    # Step 4: Test calibration (optional)
    test = input("Would you like to test the calibration? (y/n): ")
    if test.lower() == "y":
        test_calibration(camera_matrix, dist_coeffs)

    print("\n✓ Calibration complete!")
    print("\nYou can now use the calibration files for ArUco marker detection:")
    print("  - camera_calibration.npz (for Python)")
    print("  - camera_calibration.json (for other applications)")


if __name__ == "__main__":
    main()
