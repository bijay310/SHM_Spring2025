import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import easygui
import pickle
import glob

# Chessboard parameters
CHESSBOARD_SIZE = (8, 6)
CHESSBOARD_SQUARE_SIZE_MM = 25
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def extract_images_and_calibrate(video_path):
    """Extract images from the video and perform camera calibration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open the video file.")
        return None, None

    os.makedirs('calibration_images', exist_ok=True)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count, image_num = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % fps == 0:
            filename = f'calibration_images/img{image_num}.png'
            cv2.imwrite(filename, frame)
            image_num += 1
        frame_count += 1

    cap.release()

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= CHESSBOARD_SQUARE_SIZE_MM

    objpoints, imgpoints = [], []
    
    print("Extracting images complete!!")

    images = glob.glob('calibration_images/*.png')
    for image_path in images:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if len(objpoints) > 0:
        _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        with open("calibration.pkl", "wb") as f:
            pickle.dump((camera_matrix, dist_coeffs), f)
        print("Camera calibration completed. Calibration data saved to 'calibration.pkl'.")
        return camera_matrix, dist_coeffs
    else:
        print("No chessboard corners found. Calibration failed.")
        return None, None
    
    
def select_points(frame):
    """Allows the user to manually select points on the first frame."""
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([np.float32(x), np.float32(y)])
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
            cv2.putText(frame, f"({x}, {y})", (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3)

    cv2.namedWindow('Select Points to track', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Select Points to track', mouse_callback)

    while True:
        cv2.imshow('Select Points to track', frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)



def calculate_optical_flow(cap, p0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(500, 700), maxLevel=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    if not ret:
        print("Failed to read the video file.")
        return None, None, None, None, None
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    if len(p0) == 0:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=10, qualityLevel=0.2, minDistance=10, blockSize=7)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    df = pd.DataFrame()
    x = None
    y = None
    ref_frame = None
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, stt, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[stt == 1]
            good_old = p0[stt == 1]

        # draw the tracks
        x = [[] for _ in good_new] if x is None else x
        y = [[] for _ in good_new] if y is None else y
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            x[i].append(a)
            y[i].append(b)
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            cv2.putText(frame, str(i), (int(a - 25), int(b - 25)), font, 1.25, (255, 255, 255), 3)

        img = cv2.add(frame, mask)

        if ref_frame is None:
            ref_frame = img

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        frame_number += 1

    time = [frame_number / fps for frame_number in range(len(x[0]))]

    df["time/s"] = time
    for i in range(len(x)):
        df[f"x{i}"] = x[i]
        df[f"y{i}"] = y[i]

    return df, ref_frame, x, y, fps



def get_real_life_coordinates(calibration_file, tracking_data, chessboard_square_size_mm):
    """
    Calculate the real-life coordinates of tracked points using camera calibration data and tracking results.

    Parameters:
    calibration_file (str): Path to the camera calibration file (pickle file).
    tracking_data (pd.DataFrame): Dataframe containing pixel coordinates from Lucas-Kanade tracking.
    chessboard_square_size_mm (float): Size of a chessboard square in mm.

    Returns:
    pd.DataFrame: A DataFrame containing the real-life coordinates of the tracked points.
    """
    # Load camera calibration data
    with open(calibration_file, "rb") as f:
        camera_matrix, dist_coeffs = pickle.load(f)

    # Prepare real-life coordinates DataFrame
    real_life_coords = pd.DataFrame()
    real_life_coords["time/s"] = tracking_data["time/s"]

    # Extract pixel coordinates
    pixel_coordinates = []
    for i in range(len(tracking_data.columns) // 2):
        x_coords = tracking_data[f"x{i}"]
        y_coords = tracking_data[f"y{i}"]
        pixel_coordinates.append(np.stack((x_coords, y_coords), axis=-1))

    # Transform pixel coordinates to real-life coordinates
    pixel_coordinates = np.array(pixel_coordinates)
    for i, points in enumerate(pixel_coordinates):
        # Remove distortion and convert to normalized camera coordinates
        undistorted_points = cv2.undistortPoints(
            points.astype(np.float32), camera_matrix, dist_coeffs, None, camera_matrix
        ).reshape(-1, 2)

        # Convert normalized coordinates to real-world coordinates
        real_world_points = undistorted_points * (chessboard_square_size_mm / camera_matrix[0, 0])

        real_life_coords[f"x{i}_real/mm"] = real_world_points[:, 0]
        real_life_coords[f"y{i}_real/mm"] = real_world_points[:, 1]

    return real_life_coords


def main():
    video_path = easygui.fileopenbox("Select the video file for processing")
    if not video_path:
        print("No video file selected.")
        return

    # Step 1: Camera calibration
    camera_matrix, dist_coeffs = extract_images_and_calibrate(video_path)
    if camera_matrix is None or dist_coeffs is None:
        return

    # Step 2: Open video and get properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open the video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or total_frames <= 0:
        print("Invalid video properties.")
        return

    duration = total_frames / fps

    # Step 3: Prompt user for the start time for pixel tracking
    start_time = easygui.enterbox(f"Enter the start time for tracking in seconds (0 to {duration:.2f}):", 
                                  default="0")
    try:
        start_time = float(start_time)
        if start_time < 0 or start_time > duration:
            raise ValueError("Invalid time entered.")
    except ValueError:
        print("Invalid input for start time. Exiting...")
        return

    # Step 4: Skip to the desired start time
    start_frame = int(round(start_time * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video at the specified start time.")
        return

    # Step 5: Allow user to select points on the frame at the start time
    print(f"Starting pixel tracking from {start_time:.2f} seconds.")
    initial_points = select_points(first_frame)
    if len(initial_points) == 0:
        print("No points selected. Exiting...")
        return

    # Step 6: Perform optical flow tracking
    tracking_data, ref_frame, x, y, fps = calculate_optical_flow(cap, initial_points)
    if tracking_data is None:
        print("Optical flow tracking failed.")
        return

    # Step 7: Calculate real-life coordinates
    real_life_coordinates = get_real_life_coordinates("calibration.pkl", tracking_data, CHESSBOARD_SQUARE_SIZE_MM)
    print("Real-life coordinates calculation completed.")

    # Step 8: Save results
    video_title = os.path.splitext(os.path.basename(video_path))[0]
    tracking_data.to_csv(f"{video_title}_pixel_tracking.csv", index=False)
    real_life_coordinates.to_csv(f"{video_title}_real_life_coordinates.csv", index=False)
    print("Tracking results saved.")

if __name__ == "__main__":
    main()