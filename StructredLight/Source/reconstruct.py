# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys


def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")


def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(
        cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,
                                                                            0),
        fx=scale_factor,
        fy=scale_factor)
    ref_black = cv2.resize(
        cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,
                                                                            0),
        fx=scale_factor,
        fy=scale_factor)
    ref_avg = (ref_white + ref_black) / 2.0
    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region

    h, w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h, w), dtype=np.uint16)
    # print h, w

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        patt = cv2.resize(
            cv2.imread("images/pattern%03d.jpg" % (i + 2)), (0, 0),
            fx=scale_factor,
            fy=scale_factor)
        patt_gray = cv2.resize(
            cv2.imread("images/pattern%03d.jpg" %
                       (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
            fx=scale_factor,
            fy=scale_factor)
        # patt_gray = cv2.cvtColor(patt, cv2.COLOR_BGR2GRAY)
        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        # print("bit_code on_mask \n", bit_code, on_mask[i], scan_bits[i])

        for x in range(len(on_mask)):
            for y in range(len(on_mask[x])):
                if on_mask[x, y]:
                    scan_bits[x, y] = scan_bits[x, y] + bit_code
                    # TODO: populate scan_bits by putting the bit_code according to on_mask
    '''for x in range(len(scan_bits)):
                for y in range(len(scan_bits[x])):
			if scan_bits[x,y]:
                        	print scan_bits[x,y]'''
    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    # print("binary_codes_ids_codebook \n",binary_codes_ids_codebook)
    img = np.zeros((h, w, 3), np.float32)
    norm_img = np.zeros((h, w, 3), np.float32)
    colorimg = cv2.imread("images/pattern001.jpg")
    # print colorimg

    camera_points = []
    rgb_arraytemp = []
    projector_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code

            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

            (p_x, p_y) = binary_codes_ids_codebook[scan_bits[y, x]]

            if p_x >= 1279 or p_y >= 799:
                continue
            b = colorimg[y, x, 0]
            g = colorimg[y, x, 1]
            r = colorimg[y, x, 2]
            projector_points.append((p_x, p_y))
            camera_points.append((x / 2, y / 2))
            rgb_arraytemp.append((r, g, b))
            img[y, x] = [0, p_y, p_x]

    norm_img = cv2.normalize(img, norm_img, 0, 255, norm_type=cv2.NORM_MINMAX)
    projector_points = np.array(
        np.reshape(projector_points, (len(projector_points), 1, 2)),
        dtype=np.float32)
    camera_points = np.array(
        np.reshape(camera_points, (len(camera_points), 1, 2)), dtype=np.float32)
    rgb_arraytemp = np.array(
        np.reshape(rgb_arraytemp, (len(rgb_arraytemp), 1, 3)), dtype=np.float32)
    cv2.imwrite(sys.argv[1] +"correspondence.jpg", norm_img)
    print len(rgb_arraytemp)

    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl", "r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    camera_undistort = cv2.undistortPoints(camera_points, camera_K, camera_d)
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    projector_undistort = cv2.undistortPoints(projector_points, projector_K,
                                              projector_d)
    # print camera_points
    # print projector_points
    projector_matrix = np.hstack((projector_R, projector_t))
    camera_matrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    triangulatePoints = cv2.triangulatePoints(camera_matrix, projector_matrix,
                                              camera_undistort,
                                              projector_undistort).transpose()
    # print("trianPoi", triangulatePoints)
    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    points_3dTemp = cv2.convertPointsFromHomogeneous(triangulatePoints)
    # print("3dtemp ", points_3dTemp)
    print 'temp', points_3dTemp.shape
    points_3d = []
    rgb_array = []
    temp = []
    mask = (points_3dTemp[:, :, 2] > 200) & (points_3dTemp[:, :, 2] < 1400)
    # print mask
    for x in range(len(points_3dTemp)):
        for y in range(len(points_3dTemp[x])):
            # print mask[x]
            if mask[x] == True:
                points_3d.append(points_3dTemp[x, :, :])
                temp = np.concatenate(
                    (points_3dTemp[x, :, :], rgb_arraytemp[x]), axis=1)
                rgb_array.append(temp)

    # TODO: name the resulted 3D points as "points_3d"
    points_3d = np.array(points_3d)
    print 'points_3d', points_3d.shape
    print 'rgb', len(rgb_array)
    write_3d_RGB_points(rgb_array)
    return points_3d


def write_3d_RGB_points(points_3d):
    # ===== DO NOT CHANGE THIS FUNCTION =====
    print("write output point cloud color")
    #print(points_3d.shape)
    output_name = sys.argv[1] + "output_color.xyzrgb"
    # print points_3d
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d %d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2], p[0, 3],
                                             p[0, 4], p[0, 5]))


def write_3d_points(points_3d):
    # ===== DO NOT CHANGE THIS FUNCTION =====
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    # print points_3d
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))

            # return points_3d, camera_points, projector_points


if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
