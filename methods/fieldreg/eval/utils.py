import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from methods.fieldreg.eval.camera import Camera
from methods.fieldreg.eval.soccerpitch import SoccerPitch



def batch_iterate(x, batch_size):    
    batch = []
    it = iter(x)    
    while (True):
        try:
            element = next(it)
            batch.append(element)
            if (len(batch) >= batch_size):
                yield batch
                batch = []
        except (StopIteration):
            if (len(batch) > 0):
                yield batch
            return None
        
        

def distance(point1, point2):
    """
    Computes euclidian distance between 2D points
    :param point1
    :param point2
    :return: euclidian distance between point1 and point2
    """
    diff = np.array([point1['x'], point1['y']]) - np.array([point2['x'], point2['y']])
    sq_dist = np.square(diff)
    return np.sqrt(sq_dist.sum())


def mirror_labels(lines_dict):
    """
    Replace each line class key of the dictionary with its opposite element according to a central projection by the
    soccer pitch center
    :param lines_dict: dictionary whose keys will be mirrored
    :return: Dictionary with mirrored keys and same values
    """
    mirrored_dict = dict()
    for line_class, value in lines_dict.items():
        mirrored_dict[SoccerPitch.symetric_classes[line_class]] = value
    return mirrored_dict


def scale_points(points_dict, s_width, s_height):
    """
    Scale points by s_width and s_height factors
    :param points_dict: dictionary of annotations/predictions with normalized point values
    :param s_width: width scaling factor
    :param s_height: height scaling factor
    :return: dictionary with scaled points
    """
    line_dict = {}
    for line_class, points in points_dict.items():
        scaled_points = []
        for point in points:
            new_point = {'x': point['x'] * (s_width-1), 'y': point['y'] * (s_height-1)}
            scaled_points.append(new_point)
        if len(scaled_points):
            line_dict[line_class] = scaled_points
    return line_dict


def project_point(homo, p3d):
    # Project point to image space
    p3d = np.array([ p3d[0], p3d[1], 1.0 ])
    ext = homo @ p3d    
    if ext[2] < 1e-5:
        return np.zeros(3)

    p2d = ext / ext[2]
    ext = np.array([p2d[0], p2d[1], 1.0])
    return ext


def get_polylines_homo(
    homography,
    image_shape,
    sampling_factor=0.2
    ):
    """
    Given a set of camera parameters, this function adapts the camera to the desired image resolution and then
    projects the 3D points belonging to the terrain model in order to give a dictionary associating the classes
    observed and the points projected in the image.

    :param camera_annotation: camera parameters in their json/dictionary format
    :param width: image width for evaluation
    :param height: image height for evaluation
    :return: a dictionary with keys corresponding to a class observed in the image ( a line of the 3D model whose
    projection falls in the image) and values are then the list of 2D projected points.
    """
    
    height,width = image_shape

    field = SoccerPitch()
    projections = dict()
    sides = [
        np.array([1, 0, 0]),
        np.array([1, 0, -width + 1]),
        np.array([0, 1, 0]),
        np.array([0, 1, -height + 1])
    ]
    for key, points in field.sample_field_points(sampling_factor).items():
        projections_list = []
        in_img = False
        prev_proj = np.zeros(3)
        for i, point in enumerate(points):
            
            # Ignore points not on the ground
            if (point[2] != 0.0):
                continue
            
            # Project point to image space
            p3d = np.array([ point[0], point[1], 1.0 ])
            ext = project_point(homography, p3d)
            if ext[2] < 1e-5:
                # point at infinity or behind camera
                continue
                        
            if (0 <= ext[0] < width) and (0 <= ext[1] < height):

                if not in_img and i > 0:

                    line = np.cross(ext, prev_proj)
                    in_img_intersections = []
                    dist_to_ext = []
                    for side in sides:
                        intersection = np.cross(line, side)
                        if (intersection[2] < 1e-5):
                            intersection = np.ones(3)*-1
                        else:
                            intersection /= intersection[2]
                            
                        if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                            in_img_intersections.append(intersection)
                            dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
                    if in_img_intersections:
                        intersection = in_img_intersections[np.argmin(dist_to_ext)]

                        projections_list.append(
                            {
                                "x": intersection[0],
                                "y": intersection[1]
                            }
                        )

                projections_list.append(
                    {
                        "x": ext[0],
                        "y": ext[1]
                    }
                )
                in_img = True
            elif in_img:
                # first point out
                line = np.cross(ext, prev_proj)

                in_img_intersections = []
                dist_to_ext = []
                for side in sides:
                    intersection = np.cross(line, side)
                    intersection /= intersection[2]
                    if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                        in_img_intersections.append(intersection)
                        dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
                if in_img_intersections:
                    intersection = in_img_intersections[np.argmin(dist_to_ext)]

                    projections_list.append(
                        {
                            "x": intersection[0],
                            "y": intersection[1]
                        }
                    )
                in_img = False
            prev_proj = ext
        if len(projections_list):
            projections[key] = projections_list
    return projections





def get_polylines(cam, sampling_factor=0.2):
    """
    Given a set of camera parameters, this function adapts the camera to the desired image resolution and then
    projects the 3D points belonging to the terrain model in order to give a dictionary associating the classes
    observed and the points projected in the image.

    :param camera_annotation: camera parameters in their json/dictionary format
    :param width: image width for evaluation
    :param height: image height for evaluation
    :return: a dictionary with keys corresponding to a class observed in the image ( a line of the 3D model whose
    projection falls in the image) and values are then the list of 2D projected points.
    """

    width, height = cam.image_width, cam.image_height

    field = SoccerPitch()
    projections = dict()
    sides = [
        np.array([1, 0, 0]),
        np.array([1, 0, -width + 1]),
        np.array([0, 1, 0]),
        np.array([0, 1, -height + 1])
    ]
    for key, points in field.sample_field_points(sampling_factor).items():
        projections_list = []
        in_img = False
        prev_proj = np.zeros(3)
        for i, point in enumerate(points):
            ext = cam.project_point(point)
            if ext[2] < 1e-5:
                # point at infinity or behind camera
                continue
            if 0 <= ext[0] < width and 0 <= ext[1] < height:

                if not in_img and i > 0:

                    line = np.cross(ext, prev_proj)
                    in_img_intersections = []
                    dist_to_ext = []
                    for side in sides:
                        intersection = np.cross(line, side)
                        intersection /= intersection[2]
                        if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                            in_img_intersections.append(intersection)
                            dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
                    if in_img_intersections:
                        intersection = in_img_intersections[np.argmin(dist_to_ext)]

                        projections_list.append(
                            {
                                "x": intersection[0],
                                "y": intersection[1]
                            }
                        )

                projections_list.append(
                    {
                        "x": ext[0],
                        "y": ext[1]
                    }
                )
                in_img = True
            elif in_img:
                # first point out
                line = np.cross(ext, prev_proj)

                in_img_intersections = []
                dist_to_ext = []
                for side in sides:
                    intersection = np.cross(line, side)
                    intersection /= intersection[2]
                    if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                        in_img_intersections.append(intersection)
                        dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
                if in_img_intersections:
                    intersection = in_img_intersections[np.argmin(dist_to_ext)]

                    projections_list.append(
                        {
                            "x": intersection[0],
                            "y": intersection[1]
                        }
                    )
                in_img = False
            prev_proj = ext
        if len(projections_list):
            projections[key] = projections_list
    return projections


def distance_to_polyline(point, polyline):
    """
    Computes euclidian distance between a point and a polyline.
    :param point: 2D point
    :param polyline: a list of 2D point
    :return: the distance value
    """
    if 0 < len(polyline) < 2:
        dist = distance(point, polyline[0])
        return dist
    else:
        dist_to_segments = []
        point_np = np.array([point["x"], point["y"], 1])

        for i in range(len(polyline) - 1):
            origin_segment = np.array([
                polyline[i]["x"],
                polyline[i]["y"],
                1
            ])
            end_segment = np.array([
                polyline[i + 1]["x"],
                polyline[i + 1]["y"],
                1
            ])
            line = np.cross(origin_segment, end_segment)
            line /= np.sqrt(np.square(line[0]) + np.square(line[1]))

            # project point on line l
            projected = np.cross((np.cross(np.array([line[0], line[1], 0]), point_np)), line)
            projected = projected / projected[2]

            v1 = projected - origin_segment
            v2 = end_segment - origin_segment
            k = np.dot(v1, v2) / np.dot(v2, v2)
            if 0 < k < 1:

                segment_distance = np.sqrt(np.sum(np.square(projected - point_np)))
            else:
                d1 = distance(point, polyline[i])
                d2 = distance(point, polyline[i + 1])
                segment_distance = np.min([d1, d2])

            dist_to_segments.append(segment_distance)
        return np.min(dist_to_segments)


def evaluate_camera_prediction(projected_lines, groundtruth_lines, threshold):
    """
    Computes confusion matrices for a level of precision specified by the threshold.
    A groundtruth line is correctly classified if it lies at less than threshold pixels from a line of the prediction
    of the same class.
    Computes also the reprojection error of each groundtruth point : the reprojection error is the L2 distance between
    the point and the projection of the line.
    :param projected_lines: dictionary of detected lines classes as keys and associated predicted points as values
    :param groundtruth_lines: dictionary of annotated lines classes as keys and associated annotated points as values
    :param threshold: distance in pixels that distinguishes good matches from bad ones
    :return: confusion matrix, per class confusion matrix & per class reprojection errors
    """
    global_confusion_mat = np.zeros((2, 2), dtype=np.float32)
    per_class_confusion = {}
    dict_errors = {}
    detected_classes = set(projected_lines.keys())
    groundtruth_classes = set(groundtruth_lines.keys())

    false_positives_classes = detected_classes - groundtruth_classes
    for false_positive_class in false_positives_classes:
        # false_positives = len(projected_lines[false_positive_class])
        if "Circle" not in false_positive_class:
            # Count only extremities for lines, independently of soccer pitch sampling
            false_positives = 2.
        else:
            false_positives = 9.
        per_class_confusion[false_positive_class] = np.array([[0., false_positives], [0., 0.]])
        global_confusion_mat[0, 1] += 1

    false_negatives_classes = groundtruth_classes - detected_classes
    for false_negatives_class in false_negatives_classes:
        false_negatives = len(groundtruth_lines[false_negatives_class])
        per_class_confusion[false_negatives_class] = np.array([[0., 0.], [false_negatives, 0.]])
        global_confusion_mat[1, 0] += 1

    common_classes = detected_classes - false_positives_classes

    for detected_class in common_classes:

        detected_points = projected_lines[detected_class]
        groundtruth_points = groundtruth_lines[detected_class]

        per_class_confusion[detected_class] = np.zeros((2, 2))

        all_below_dist = 1
        for point in groundtruth_points:

            dist_to_poly = distance_to_polyline(point, detected_points)
            if dist_to_poly < threshold:
                per_class_confusion[detected_class][0, 0] += 1
            else:
                per_class_confusion[detected_class][0, 1] += 1
                all_below_dist *= 0

            if detected_class in dict_errors.keys():
                dict_errors[detected_class].append(dist_to_poly)
            else:
                dict_errors[detected_class] = [dist_to_poly]

        if all_below_dist:
            global_confusion_mat[0, 0] += 1
        else:
            global_confusion_mat[0, 1] += 1

    return global_confusion_mat, per_class_confusion, dict_errors


def get_accuracy_at_threshold(lines_gt, lines_pred, threshold=5):
    confusion1, per_class_conf1, reproj_errors1 = evaluate_camera_prediction(
            lines_pred, lines_gt, threshold
        )

    confusion2, per_class_conf2, reproj_errors2 = evaluate_camera_prediction(
            lines_pred, mirror_labels(lines_gt), threshold
        )
        
    accuracy1, accuracy2 = 0., 0.
    if confusion1.sum() > 0:
        accuracy1 = confusion1[0, 0] / confusion1.sum()

    if confusion2.sum() > 0:
        accuracy2 = confusion2[0, 0] / confusion2.sum()

    if accuracy1 > accuracy2:
        accuracy = accuracy1
        confusion = confusion1
        per_class_conf = per_class_conf1
        reproj_errors = reproj_errors1
    else:
        accuracy = accuracy2
        confusion = confusion2
        per_class_conf = per_class_conf2
        reproj_errors = reproj_errors2

    # Return results
    return accuracy, confusion

