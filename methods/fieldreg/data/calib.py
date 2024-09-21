
from typing import Tuple
import torch
import numpy as np
import cv2
import math


def fov_to_f(fov):
    return 1.0 / (2.0 * math.tan(fov/2.0))

def f_to_fov(f, x):
    return 2.0 * math.atan2(x, (2.0*f))


class Calibrator:
    def __init__(
        self,
        playfield,
        threshold=0.9
    ):
        self.playfield = playfield
        self.threshold = threshold

        # Center of the playing field
        self.center = np.array(
            [ playfield.length / 2.0, playfield.width / 2.0 ],
            dtype=np.float32
        )

    def find_homography(
        self,
        target_shape,
        dmaps: torch.Tensor
    ):
        """ Extract lines and intersections from distance maps
            and find homography matrix
        """

        dmaps = dmaps.detach().cpu().numpy()
        dmaps = np.transpose(dmaps, axes=(1,2,0))

        TH, TW = target_shape
        DH, DW = dmaps.shape[0], dmaps.shape[1]
        scale = np.array([ TW/DW, TH/DH ], dtype=np.float32)

        # Extract points
        pts_3d, pts_2d = extract_points(
            dmaps=dmaps, 
            playfield=self.playfield,
            threshold=self.threshold
        )        

        # Find homography from point correspondences
        if (pts_2d.shape[1] >= 4):
            pts_3d = pts_3d + self.center
            homography, _ = cv2.findHomography(
                np.multiply(pts_2d, scale).reshape(-1, 1, 2), 
                pts_3d.reshape(-1, 1, 2), 
                cv2.RANSAC, 
                10
            )
        else:
            homography = None

        image_keypoints = np.zeros(shape=(DH, DW), dtype=np.uint8)

        pts_2d = np.rint(pts_2d).astype(np.int32)
        for i in range(pts_2d.shape[1]):
            px = pts_2d[0,i,0]
            py = pts_2d[0,i,1]
            if (0 <= px < DW) and (0 <= py < DH):
                image_keypoints[py, px] = 1        

        return homography, image_keypoints


def load_calib360_image(
    basename: str,
    grayscale: bool=False
):
    cam_filepath = basename + "-cam.png"

    # read main image
    try:
        if (grayscale):
            main_image = cv2.imread(cam_filepath, cv2.IMREAD_COLOR)
            main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
            main_image = np.expand_dims(main_image, axis=2)
            main_image = np.repeat(main_image, 3, axis=2)
        else:
            main_image = cv2.imread(cam_filepath, cv2.IMREAD_COLOR)
            main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)

    except:
        raise Exception(f"Failed to load image: {cam_filepath}")

    return main_image


def load_calib360_dmaps(
    basename: str, 
    num_channels: int
):
    # read line images
    images = []
    channels_per_file = 3
    num_files = (num_channels + (channels_per_file - 1)) // channels_per_file
    for i in range(num_files):
        image_filepath = basename + f"-lines{i}.png"
        try:
            line_image = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
            line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

            images.append(line_image)
        except:
            raise Exception(f"Failed to load image: {image_filepath}")

    # concat all line images, and select just the necessary number of channels
    lines = np.concatenate(images, axis=2)
    lines = lines[:,:,0:num_channels]

    # convert to float
    lines = lines.astype(float) / 255.0
    return lines





def load_calib360_image_pair(
    basename: str, 
    num_channels: int,
    grayscale: bool=False
) -> Tuple[np.ndarray, np.ndarray]:
    """ Loads the main camera image and all associated line images.
    Returns the tuple, where the first element is the main image,
    the second image is the concatenated list of line images.
    """
    cam_filepath = basename + "-cam.png"

    # read main image
    try:
        if (grayscale):
            main_image = cv2.imread(cam_filepath, cv2.IMREAD_COLOR)
            main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
            main_image = np.expand_dims(main_image, axis=2)
            main_image = np.repeat(main_image, 3, axis=2)
        else:
            main_image = cv2.imread(cam_filepath, cv2.IMREAD_COLOR)
            main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)

    except:
        raise Exception(f"Failed to load image: {cam_filepath}")

    # read line images
    images = []
    channels_per_file = 3
    num_files = (num_channels + (channels_per_file - 1)) // channels_per_file
    for i in range(num_files):
        image_filepath = basename + f"-lines{i}.png"
        try:
            line_image = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
            line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

            images.append(line_image)
        except:
            raise Exception(f"Failed to load image: {image_filepath}")

    # concat all line images, and select just the necessary number of channels
    lines = np.concatenate(images, axis=2)
    lines = lines[:,:,0:num_channels]

    # convert to float
    lines = lines.astype(float) / 255.0

    return main_image, lines


def get_thresholded_lines(lines, threshold=0.95):
    # <H,W,C> -> <C,H,W> -> <C, H*W>
    H,W,C = lines.shape
    lines_t = np.transpose(lines, axes=(2,0,1))
    lines_f = lines_t.reshape((C,-1))
    max_value = lines_f.max(axis=1, keepdims=True)
    thresholded = (lines_f > (max_value*threshold)).astype(np.uint8)*255
    # Go back to the original image shape, but transposed
    thresholded = thresholded.reshape((C,H,W))
    thresholded = np.transpose(thresholded, axes=(1,2,0))
    return thresholded.copy()

  
def get_local_maxima(img, threshold=0.9):
    # img has shape <H,W,C>
    kernel = np.ones((3, 3), np.uint8) 
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    relevant_mask = ((img > threshold) * 1).astype(np.uint8)
    max_mask = ((dilated_img == img) * 1).astype(np.uint8)
    max_img = img * max_mask * relevant_mask
    return max_img

def detect_lines_per_channel(image_lines, threshold=0.9):
    maxima = get_local_maxima(image_lines, threshold)
    nch = maxima.shape[2]
    result = []
    for ch in range(nch):

        # Get positions of all points of the local maxima
        single_channel = maxima[:,:,ch]
        points = np.argwhere(single_channel > 0)
        if (points.shape[0] > 10):
            points[:,[1,0]]=points[:,[0,1]]
            w = image_lines.shape[1]
            vx,vy,cx,cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.1, 0.1)
            result.append(
                (ch, 
                    [(
                        np.array([cx[0]+vx[0]*w, cy[0]+vy[0]*w], dtype=np.float32),
                        np.array([cx[0]-vx[0]*w, cy[0]-vy[0]*w], dtype=np.float32),
                    )]                 
                 )
            )
        else:
            result.append((ch, []))

    return result



def _intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def _line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def _get_intersection(p1, p2, p3, p4):
    line1 = _line(p1, p2)
    line2 = _line(p3, p4)
    R = _intersection(line1, line2)
    if R:
        return True, R
    return False, (0.0, 0.0)

def _get_intersection_points(hgroup, vgroup):
    result = []
    for idx_i, line_segments_i in hgroup:
        if len(line_segments_i) == 0: continue

        for idx_j, line_segments_j in vgroup:
            if len(line_segments_j) == 0: continue
            idx_j -= len(hgroup)

            # Each line is defined by two points
            p1_i, p2_i = line_segments_i[0]
            p1_j, p2_j = line_segments_j[0]
            
            # Find intersection point
            found, point = _get_intersection(p1_i, p2_i, p1_j, p2_j)
            if (found):
                result.append({ "i,j": (idx_i, idx_j), "p": point})

    return result


def zxz_from_rotation(R):
    st = math.sqrt(R[0,2]*R[0,2] + R[1,2]*R[1,2])
    singular = st < 1e-6
    if not singular:
        p = math.atan2(R[2,0], -R[2,1])
        r = math.atan2(R[0,2], R[1,2])
        t = math.atan2(st, R[2,2])
    else:
        p = 0
        r = 0
        t = 0
    return np.array([p,t,r])



def calibrate_from_correspondences(
    points2d, points3d,
    image_size
):
    H,W = image_size
    points_3d = np.array(points3d).astype(np.float32)
    points_2d = np.array(points2d).astype(np.float32)

    # Preco preboha !!!
    #points_2d = np.round(50.0*points_2d)/50.0

    # Solve PnP
    dist_coeffs = np.zeros((4,1))
    K = np.array([
        (W, 0, W/2),
        (0, W, H/2),
        (0, 0, 1)
        ])

    flags = 0 + \
        cv2.CALIB_USE_INTRINSIC_GUESS + \
        cv2.CALIB_FIX_ASPECT_RATIO + \
        cv2.CALIB_ZERO_TANGENT_DIST + \
        cv2.CALIB_FIX_PRINCIPAL_POINT + \
        cv2.CALIB_FIX_K1 + \
        cv2.CALIB_FIX_K2 + \
        cv2.CALIB_FIX_K3 + \
        cv2.CALIB_FIX_K4 + \
        cv2.CALIB_FIX_K5 + \
        cv2.CALIB_FIX_K6
    
    # Not enough points!
    if (len(points_3d) < 6):
        return None

    # termination criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 
        50, 
        0.00001
    )    
    
    ret, camera_mat, distortion, rvecs, tvecs = cv2.calibrateCamera( 
        objectPoints=[points_3d],
        imagePoints=[points_2d],
        imageSize=(H,W), 
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        flags=flags,
        criteria=criteria
        ) 
    
    #np_rodrigues = np.asarray(rvecs[:,:],np.float64)
    rmat = cv2.Rodrigues(rvecs[0])[0]
    T = tvecs[0]
    
    MV = np.array([
        (rmat[0,0], rmat[0,1], rmat[0,2], T[0,0]),
        (rmat[1,0], rmat[1,1], rmat[1,2], T[1,0]),
        (rmat[2,0], rmat[2,1], rmat[2,2], T[2,0]),
        (0, 0, 0, 1)
    ], dtype=np.float64)
    vm = np.linalg.inv(MV)

    camera_position = np.array([ vm[0,3], vm[1,3], vm[2,3] ])
    
    angles = zxz_from_rotation(MV[0:3,0:3])    
    angles = angles * 180/math.pi    
    f = camera_mat[1,1]
    fv = f_to_fov(f, H) * 180 / math.pi
    fh = f_to_fov(f, W) * 180 / math.pi

    result = {
        "position": camera_position,
        "angles": angles,
        "fov_v": fv,
        "fov_h": fh,
        "mv": MV
    }
    return result


def extract_points(
    dmaps,
    playfield,
    threshold=0.9
):
    """ Extract lines, and intersection points
    """

    # Make sure we have the right number of lines
    hcount, vcount = playfield.grid_shape
    #assert(lines.shape[2] == (hcount + vcount), "Line count is incorrect!")

    detected_lines = detect_lines_per_channel(dmaps, threshold=threshold)        

    # Extract intersection points
    hgroup = detected_lines[:hcount]
    vgroup = detected_lines[hcount:]
    points = _get_intersection_points(hgroup, vgroup)

    # Extract point correspondences
    points_3d_list = []
    points_2d_list = []
    for p in points:
        points_3d_list.append(
            (            
                playfield.points[p["i,j"]][0],
                playfield.points[p["i,j"]][1]
            )
        )
        points_2d_list.append(p["p"])

    pts_3d = np.array([ points_3d_list ], dtype=np.float32)
    pts_2d = np.array([ points_2d_list ], dtype=np.float32)
    return pts_3d, pts_2d


def extract_camera_pose(
    image_pair, 
    playfield,
    threshold=0.9
):
    """ Extract lines, and intersection points
    """

    main_image, lines = image_pair
    H,W = main_image.shape[0], main_image.shape[1]

    # Make sure we have the right number of lines
    hcount, vcount = playfield.grid_shape
    #assert(lines.shape[2] == (hcount + vcount), "Line count is incorrect!")

    detected_lines = detect_lines_per_channel(lines, threshold=threshold)        

    # Extract intersection points
    hgroup = detected_lines[:hcount]
    vgroup = detected_lines[hcount:]
    points = _get_intersection_points(hgroup, vgroup)

    # Extract point correspondences
    points_3d_list = []
    points_2d_list = []
    for p in points:
        points_3d_list.append(playfield.points[p["i,j"]])
        points_2d_list.append(p["p"])
        
        
    result = calibrate_from_correspondences(
        points2d=points_2d_list,
        points3d=points_3d_list,
        image_size=(H,W)
    )
    if (result is not None):
        result["lines"] = detected_lines
        result["points"] = points
        result["p3d"] = points_3d_list
        result["p2d"] = points_2d_list    

    return result

