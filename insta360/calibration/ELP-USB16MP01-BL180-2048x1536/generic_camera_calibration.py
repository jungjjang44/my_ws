import cv2
import yaml
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import os
import glob
import math
import random
import json
import pandas as pd
import argparse
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--visualize', action="store_true")
args = parser.parse_args()

def display_images(images, max_num=8):
    # Number of images
    images = images[:max_num]
    N = len(images)

    # Determine the grid shape based on the number of images
    if N % 4 == 0:
        nrows, ncols = N // 4, 4
    else:
        nrows, ncols = N // 4 + 1, 4
    
    # Create a figure with subplots in a grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing

    # Load and display each image
    for idx, image in enumerate(images):
        axes[idx].imshow(image)
        axes[idx].axis('off')  # Turn off axis

    # If the number of images is odd, hide the last subplot
    if N % 4 != 0:
        axes[-1].axis('off')

    plt.tight_layout()
    plt.show()

def save_images(images, image_names, reprojection_errors, save_path, color_convert=True, extension='png', log=False):
    os.makedirs(save_path, exist_ok=True)
    # Loop through the images and their corresponding names
    for img, name, error in zip(images, image_names, reprojection_errors):
        # Build the full file path
        name = name.rstrip(".png").rstrip(".jpg")
        full_path = os.path.join(save_path, f"{name}_{str(round(error, 4))}.{extension}")
        # Save the image
        if color_convert:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(full_path, img)
        if log:
            print(f"Saved: {full_path}")  # Optional: confirmation message
            
def main():    
    ######################### Calibration Setting #########################
    print("\nStep 1. Calibration Setting\n")
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
            
    pprint.pprint(config)
    camera_name = config["camera_name"]
    base_path = config["base_path"]
    square_size = config["square_size"]
    HFoV = config["HFoV"]
    VFoV = config["VFoV"]
    num_checkboard_width_intersection = config["num_checkboard_width_intersection"]
    num_checkboard_height_intersection = config["num_checkboard_height_intersection"]
    extrinsic = config["extrinsic"]
    
    ######################### Input File Check ############################
    assert os.path.exists(base_path + os.sep  + "Intrinsic") == True
    assert os.path.exists(base_path + os.sep  + "Intrinsic" + os.sep + "ORIGIN") == True
    
    if extrinsic:
        assert os.path.exists(base_path + os.sep  + "Extrinsic") == True
        assert os.path.exists(base_path + os.sep  + "Extrinsic" + os.sep + camera_name + "_EXTRINSIC.png") == True
        assert os.path.exists(base_path + os.sep  + "Extrinsic" + os.sep + camera_name) == True
        assert os.path.exists(base_path + os.sep  + "Extrinsic" + os.sep + camera_name + os.sep + "points.csv") == True
    #######################################################################
    
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    
    _img_shape = None
    image_paths = glob.glob(base_path.rstrip(os.sep) + os.sep + "Intrinsic" + os.sep + "ORIGIN" + os.sep + "*.png")
    image_paths.sort()
    print(">>> The number of images to get intrinsic parameter: ", len(image_paths))

    img = cv2.cvtColor(cv2.imread(image_paths[0]), cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = img.shape
    print(f">>> WIDTH: {image_width}, HEIGHT: {image_height}")
    
    ######################### Setting Intrinsic Checkboard Images ##############################
    print("\nStep 2. Setting Intrinsic Checkboard Images\n")
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    valid_image_names = []
    valid_images = []
    valid_visual_images = []
    for index, image_path in enumerate(image_paths):    
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        is_found = False
        '''
        - search checkboard images with 'findChessboardCorners' function.
        - chckboard size is ('num_checkboard_width_intersection', 'num_checkboard_height_intersection')
        - if checkboard is not found, search smaller size of checkboard with delta_x, delta_y
        '''
        for delta_x in [0, -1, -2, -3, -4, -5]:
            for delta_y in [0, -1, -2, -3, -4, -5]:
                checkboard = (num_checkboard_width_intersection+delta_x, num_checkboard_height_intersection+delta_y)
                objp = np.zeros((1, checkboard[0]*checkboard[1], 3), np.float32)
                objp[0,:,:2] = np.mgrid[0:checkboard[0], 0:checkboard[1]].T.reshape(-1, 2)
                objp *= square_size
                
                ret, corners = cv2.findChessboardCorners(gray, checkboard, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
                # If found, add object points, image points (after refining them)
                
                if ret == True:
                    is_found = True
                    print(checkboard, image_path)
                    objpoints.append(objp)
                    corners_pixel = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                    imgpoints.append(corners)
            
                    valid_images.append(img)
                    visual_img = cv2.drawChessboardCorners(img, checkboard, corners_pixel, ret)
                    valid_visual_images.append(visual_img)
                    valid_image_names.append(os.path.basename(image_path))
                    break
                    
            if is_found:
                break
        
        # remove unnecessary image files which interrupts calibration.
        if is_found == False:
            os.remove(image_path)
            
    objpoints_copy = objpoints.copy()
    imgpoints_copy = imgpoints.copy()
    image_paths_copy = image_paths.copy()
    valid_images_copy = valid_images.copy()
    valid_visual_images_copy = valid_visual_images.copy()
    
    if args.visualize:
        display_images(valid_visual_images)
        
    
    if extrinsic:
        ######################### Add Extrinsic Calibration #########################
        print("\nStep 3. Add Extrinsic Calibration\n")
        extrinsic_base_path = base_path + os.sep + "Extrinsic"
        extrinsic_image_path = extrinsic_base_path + os.sep + camera_name + "_EXTRINSIC.png"
        extrinsic_image = cv2.cvtColor(cv2.imread(extrinsic_image_path), cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = extrinsic_image.shape
        print(f">>> WIDTH: {image_width}, HEIGHT: {image_height}")
        
        df = pd.read_csv(extrinsic_base_path + os.sep + camera_name + os.sep + "points.csv")
        extrinsic_objp = np.zeros((1, len(df), 3)).astype(np.float32)
        extrinsic_corners = np.zeros((len(df), 1, 2)).astype(np.float32)

        for index, row in df.iterrows():
            u = int(row["u"])
            v = int(row["v"])
            x = float(row["X"])
            y = float(row["Y"])
            z = float(row["Z"])

            extrinsic_corners[index][0][0] = u
            extrinsic_corners[index][0][1] = v
            extrinsic_objp[0][index][0] = x
            extrinsic_objp[0][index][1] = y
            extrinsic_objp[0][index][2] = z

        extrinsic_visual_image = extrinsic_image.copy()
        for i in range(len(extrinsic_corners)):
            u, v = extrinsic_corners[i][0]
            u = int(u)
            v = int(v)
            cv2.circle(extrinsic_visual_image, (u, v), 3, (255, 0, 0), thickness=-1)

        if args.visualize:
            plt.imshow(extrinsic_visual_image)
            plt.show()
    else:
        print("\nSkipped: (Step 3. Add Extrinsic Calibration)\n")

    # Print the lengths of the lists
    print("#imgpoints: ", len(imgpoints))
    print("#objpoints: ", len(objpoints))
    print("#valid_images: ", len(valid_images))
    print("#valid_visual_images: ", len(valid_visual_images))
    
    
    ######################### Calibrate Intrinsic & Extrinsic Parameters ###########################
    print("\nStep 4. Calibrate Intrinsic & Extrinsic Parameters\n")
    objpoints = objpoints_copy.copy()
    imgpoints = imgpoints_copy.copy()
    image_paths = image_paths_copy.copy()
    valid_images = valid_images_copy.copy()
    valid_visual_images = valid_visual_images_copy.copy()

    N_OK = len(objpoints)
    indices = list(range(N_OK))

    # Shuffle the indices
    random.shuffle(indices)    
    objpoints = [objpoints[i] for i in indices]
    imgpoints = [imgpoints[i] for i in indices]
    image_paths = [image_paths[i] for i in indices]
    valid_images = [valid_images[i] for i in indices]
    valid_visual_images = [valid_visual_images[i] for i in indices]

    if extrinsic:
        N_OK += 1
        objpoints.append(extrinsic_objp)
        imgpoints.append(extrinsic_corners)
        image_paths.append(extrinsic_image_path)
        valid_images.append(extrinsic_image)
        valid_visual_images.append(extrinsic_visual_image)

    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    DIM = (image_width, image_height)
    # Calculate the focal lengths based on the field of view
    init_fx = image_width / (2 * math.tan(math.radians(HFoV/2 - 10)))
    init_fy = image_height / (2 * math.tan(math.radians(VFoV/2 - 10)))

    # Principal point assumed to be at the center of the image
    init_cx = image_width / 2
    init_cy = image_height / 2

    K = np.zeros((3, 3))    
    K[0][0] = init_fx
    K[1][1] = init_fy
    K[0][2] = init_cx
    K[1][2] = init_cy

    while True:    
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    objpoints, # objectPoints vector of vectors of calibration pattern points in the calibration pattern
                    imgpoints, # imagePoints vector of vectors of the projections of calibration pattern points.
                    (image_width, image_height),
                    K,
                    D,
                    rvecs,
                    tvecs,
                    calibration_flags,
                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
            break
        except cv2.error as err:
            try:            
                idx = int(err.msg.split('array ')[1][0])  # Parse index of invalid image from error message
                objpoints.pop(idx)
                imgpoints.pop(idx)
                image_paths.pop(idx)
                valid_images.pop(idx)
                valid_visual_images.pop(idx)
                rvecs.pop(idx)
                tvecs.pop(idx)
                print("Removed ill-conditioned image {} from the data.  Trying again... {} images remain".format(idx, len(objpoints)))
                print("Remove unnecessary images to calibrate before running code. It's better to calibrate this.")
            except IndexError:
                print(err.msg)
                continue

    print(">>> Found " + str(N_OK) + " valid images for calibration")
    print(">>> DIM = " + str(_img_shape[::-1]))
    print(">>> K = \n", K)
    print(">>> D = \n", D)
    
    ################################# Check Intrinsic & Extrinsic Parameter Qualities with Reprojection. ################################ 
    print("\nStep 5. Check Intrinsic & Extrinsic Parameter Qualities with Reprojection.\n")
    N_OK = len(objpoints)
    reprojection_errors = []
    for i in range(N_OK):
        reprojection_points, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        reprojection_points = reprojection_points[0]
        # reprojection_points, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, D)
        reprojection_error = cv2.norm(imgpoints[i][:, 0, :], reprojection_points, cv2.NORM_L2)/len(reprojection_points)
        reprojection_errors.append(reprojection_error)
        for pt in reprojection_points:
            cv2.circle(valid_images[i], (int(pt[0]), int(pt[1])), 3, (255,0,0), -1)

    print("Total error: ", sum(reprojection_errors) / len(reprojection_errors))
    if args.visualize:
        display_images(valid_images)
    save_images(valid_visual_images, valid_image_names, reprojection_errors, base_path + os.sep + "Intrinsic" + os.sep + "REPROJECTION")
    
    if args.visualize:
        # Define the polynomial coefficients
        coefficients = [1.0] + np.round(D, 5).reshape(-1).tolist()
        coefficients_9th = [item for subitem in zip(coefficients, [0]*(len(coefficients)-1)) for item in subitem] + [coefficients[-1]]

        # Create a polynomial function using numpy's poly1d
        p = np.poly1d(coefficients_9th)

        # Generate x values
        x = np.linspace(0, np.pi/2, 50)

        # Compute y values
        y = p(x)

        # Plotting
        plt.figure(figsize=(8, 6))  # Set the figure size
        plt.plot(x, y)
        plt.title('Nineth Degree Polynomial')  # Title of the plot
        plt.xlabel('x')  # X-axis label
        plt.ylabel('y')  # Y-axis label
        plt.grid(True)  # Turn on the grid
        plt.axhline(0, color='black',linewidth=0.5)  # Add a horizontal line at y=0
        plt.axvline(0, color='black',linewidth=0.5)  # Add a vertical line at x=0
        plt.show()  # Display the plot
    
    ################################# Save Intrinsic Parameters. ################################ 
    print("\nStep 6. Save Intrinsic Parameters.\n")
    intrinsic_result_dict = {}
    intrinsic_result_dict["intrinsic"] = np.round(K, 5).reshape(-1).tolist()
    intrinsic_result_dict["distortion"] = [1.0] + np.round(D, 5).reshape(-1).tolist()
    intrinsic_result_dict["reprojection_error"] = round(sum(reprojection_errors) / len(reprojection_errors), 5)
    with open(base_path.rstrip(os.sep) + os.sep + "Intrinsic" + os.sep + "intrinsic.json", "w") as file:
        json.dump(intrinsic_result_dict, file, indent=4, separators=(',', ': '))
    print(f">>> saved to {os.path.dirname(base_path).rstrip(os.sep) + os.sep + 'intrinsic.json'}")
    
    
    ################################# Get Sensor Position. ################################ 
    if extrinsic:
        rvec = rvecs[-1]
        tvec = tvecs[-1]

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec

        # Create a 4x4 identity matrix and insert R and t
        Rt = np.zeros((4, 4)).astype(np.float32)
        Rt[:3, :3] = R
        Rt[:3, 3] = t.reshape(-1)
        Rt[3, 3] = 1.0

        print("\nR: \n", R)
        print("\nt: \n", t)
        print("\nRt: \n", Rt)
        
        CameraPosition = -R.T @ t
        print(f"camera position: \nX:{CameraPosition[0]}\nY:{CameraPosition[1]}\nZ:{CameraPosition[2]}")
        
    ################################# Save Extrinsic & Intrinsic Calibration Result. ################################ 
    if extrinsic:
        # Create a dictionary to store the calibration data
        calibration_dict = {}
        calibration_dict[camera_name] = {}
        calibration_dict[camera_name]["Position"] = np.round(CameraPosition, 5).reshape(-1).tolist()

        calibration_dict[camera_name]["Intrinsic"] = {}
        calibration_dict[camera_name]["Intrinsic"]["K"] = np.round(K, 5).reshape(-1).tolist()
        calibration_dict[camera_name]["Intrinsic"]["D"] = [1.0] + np.round(D, 5).reshape(-1).tolist()
        calibration_dict[camera_name]["Intrinsic"]["ReprojectionError"] = round(sum(reprojection_errors) / len(reprojection_errors), 5)

        # R_w->c @ P_w + t_w->c = P_c
        # (R_w->c)^{1} @ (P_c - t_w->c) = P_w
        # R_c->w = (R_w->c)^{-1} = (R_w->c)^{T}
        # t_c->w = -(R_w->c)^{-1} @ t_w->c
        calibration_dict[camera_name]["Extrinsic"] = {}
        calibration_dict[camera_name]["Extrinsic"]["World"] = {}
        calibration_dict[camera_name]["Extrinsic"]["World"]["Camera"] = {}
        calibration_dict[camera_name]["Extrinsic"]["World"]["Camera"]["R"] = np.round(R, 5).reshape(-1).tolist()
        calibration_dict[camera_name]["Extrinsic"]["World"]["Camera"]["t"] = np.round(t, 5).reshape(-1).tolist()

        calibration_dict[camera_name]["Extrinsic"]["Camera"] = {}
        calibration_dict[camera_name]["Extrinsic"]["Camera"]["World"] = {}
        calibration_dict[camera_name]["Extrinsic"]["Camera"]["World"]["R"] = np.round(R.T, 5).reshape(-1).tolist()
        calibration_dict[camera_name]["Extrinsic"]["Camera"]["World"]["t"] = np.round(-R.T @ t, 5).reshape(-1).tolist()

        # Pretty print the calibration dictionary        
        print("Intrinsic & Extrinsic Calibration Results: \n")
        pprint.pprint(calibration_dict)

        # Save the dictionary to a JSON file
        file_path = os.path.join(base_path.rstrip(os.sep) + os.sep + camera_name + "_calibration.json")
        with open(file_path, "w") as file:
            json.dump(calibration_dict, file, indent=4, separators=(',', ': '))

        

if __name__ == "__main__":
    main()
