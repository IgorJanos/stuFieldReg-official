import cv2
import numpy as np
from typing import List, Any
import math



# Lines are represented as (startpoint, endpoint)

class Playfield:
    def __init__(self, length=105, width=68, grid_shape=(5,5)):
        self.length = length
        self.width = width
        self.grid_shape = grid_shape
        self.hlines, self.vlines = self._build_lines()
        self.points = self._build_known_points(self.hlines, self.vlines)

    def get_distance_map(self, image_shape=(360, 640), radius=1.0):
        H, W = image_shape
        p0 = get_pixel_coords(image_shape)
        p0 = self._to_field_coordinates(p0)

        maps = []
        all_lines = self.hlines + self.vlines

        for line in all_lines:
            p1,p2 = line[0], line[1]
            p1 = p1.reshape(2,1,1)
            p2 = p2.reshape(2,1,1)

            # Distance of P1, P2
            dp1p2 = np.sqrt(np.sum((p2 - p1) ** 2))

            # Distance map
            distance_map = np.abs(
                (p2[0] - p1[0])*(p1[1] - p0[1]) -
                (p1[0] - p0[0])*(p2[1] - p1[1])
            ) / dp1p2

            # Into proper range
            distance_map = 1.0 - np.clip((distance_map/radius), 0, 1)            
            maps.append(distance_map)

        return maps

    def _to_field_coordinates(self, pixcoords):
        _,H,W = pixcoords.shape
        FL,FW = self.length, self.width

        result = np.zeros(shape=pixcoords.shape, dtype=np.float32)
        result[0] = pixcoords[1]*FL/W - FL/2.0
        result[1] = pixcoords[0]*FW/H - FW/2.0
        return result


    def _build_lines(self):
        hlines = []
        vlines = []

        sl = self.length / self.grid_shape[0]
        sw = self.width / self.grid_shape[1]
        c = np.array([self.length / 2.0, self.width / 2.0])

        sx = np.arange(self.grid_shape[0]) * sl
        sy = np.arange(self.grid_shape[1]) * sw

        hlines = [(
                    np.array([x + sl/2, 0]) - c, 
                    np.array([x + sl/2, self.width]) - c
                ) 
                for x in sx
                ]
        vlines = [(
                    (np.array([0, y + sw/2]) - c) * np.array([1, 1]), 
                    (np.array([self.length, y + sw/2]) - c) * np.array([1, 1])
                ) 
                for y in sy
                ]
        
        return hlines, vlines
    
    def _build_known_points(self, hlines, vlines):
        points = {}

        for idx_i, hline in enumerate(hlines):
            for idx_j, vline in enumerate(vlines):

                # Coordinates of the intersect point
                x = hline[0][0]
                y = vline[0][1]
                z = 0

                key = (idx_i,idx_j)
                points[key] = (x,y,z)

        return points
    
    def draw_playing_field(self, shape=(2048,2048)):
        H,W = shape
        result = np.zeros(shape=(H,W,3), dtype=np.uint8)

        class Mapper:
            def __init__(
                self, 
                image_width, 
                image_height,
                playingfield_length,
                playingfield_width
                ):
                self.IW = image_width
                self.IH = image_height
                self.PL = playingfield_length
                self.PW = playingfield_width

                self.cx = image_width / 2
                self.cy = image_height / 2

            def to_xy(self, p):
                x = self.cx + p[0] * self.IW / self.PL
                y = self.cy + p[1] * self.IH / self.PW
                return x,y
            
            def to_xy_int(self, p):
                x,y = self.to_xy(p)
                return int(x), int(y)
            

        m = Mapper(W,H,self.length,self.width)
        c = (255,0,0)
        lw = 3

        GOAL_LINE_TO_PENALTY_MARK = 11.0
        PENALTY_AREA_WIDTH = 40.32
        PENALTY_AREA_LENGTH = 16.5
        GOAL_AREA_WIDTH = 18.32
        GOAL_AREA_LENGTH = 5.5
        CENTER_CIRCLE_RADIUS = 9.15
        GOAL_HEIGHT = 2.44
        GOAL_LENGTH = 7.32    

        XL = -self.length / 2.0
        XR = self.length / 2.0
        XC = 0
        YF = self.width / 2.0
        YN = -self.width / 2.0
        YC = 0

        # Draw
        cv2.line(result, m.to_xy_int((XL, YF, 0)), m.to_xy_int((XR, YF, 0)), color=c, thickness=lw) 
        cv2.line(result, m.to_xy_int((XL, YF, 0)), m.to_xy_int((XL, YN, 0)), color=c, thickness=lw) 
        cv2.line(result, m.to_xy_int((XL, YN, 0)), m.to_xy_int((XR, YN, 0)), color=c, thickness=lw) 
        cv2.line(result, m.to_xy_int((XR, YN, 0)), m.to_xy_int((XR, YF, 0)), color=c, thickness=lw) 

        # Penalty area - left
        cv2.line(result, 
                 m.to_xy_int((XL,                       YC - PENALTY_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XL + PENALTY_AREA_LENGTH, YC - PENALTY_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 
        cv2.line(result, 
                 m.to_xy_int((XL,                       YC + PENALTY_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XL + PENALTY_AREA_LENGTH, YC + PENALTY_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 
        cv2.line(result, 
                 m.to_xy_int((XL + PENALTY_AREA_LENGTH, YC - PENALTY_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XL + PENALTY_AREA_LENGTH, YC + PENALTY_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 
        
        # Goal area - left
        cv2.line(result, 
                 m.to_xy_int((XL,                    YC - GOAL_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XL + GOAL_AREA_LENGTH, YC - GOAL_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 
        cv2.line(result, 
                 m.to_xy_int((XL,                    YC + GOAL_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XL + GOAL_AREA_LENGTH, YC + GOAL_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 
        cv2.line(result, 
                 m.to_xy_int((XL + GOAL_AREA_LENGTH, YC - GOAL_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XL + GOAL_AREA_LENGTH, YC + GOAL_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 


        # Penalty area - right
        cv2.line(result, 
                 m.to_xy_int((XR,                       YC - PENALTY_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XR - PENALTY_AREA_LENGTH, YC - PENALTY_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 
        cv2.line(result, 
                 m.to_xy_int((XR,                       YC + PENALTY_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XR - PENALTY_AREA_LENGTH, YC + PENALTY_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 
        cv2.line(result, 
                 m.to_xy_int((XR - PENALTY_AREA_LENGTH, YC - PENALTY_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XR - PENALTY_AREA_LENGTH, YC + PENALTY_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 

        # Goal area - right
        cv2.line(result, 
                 m.to_xy_int((XR,                    YC - GOAL_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XR - GOAL_AREA_LENGTH, YC - GOAL_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 
        cv2.line(result, 
                 m.to_xy_int((XR,                    YC + GOAL_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XR - GOAL_AREA_LENGTH, YC + GOAL_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 
        cv2.line(result, 
                 m.to_xy_int((XR - GOAL_AREA_LENGTH, YC - GOAL_AREA_WIDTH/2.0, 0)), 
                 m.to_xy_int((XR - GOAL_AREA_LENGTH, YC + GOAL_AREA_WIDTH/2.0, 0)), 
                 color=c, thickness=lw) 

        # Center
        num_sides = 200
        for i in range(num_sides):
            a1 = 2*math.pi * (i+0)/num_sides
            a2 = 2*math.pi * (i+1)/num_sides
            x1 = CENTER_CIRCLE_RADIUS * math.sin(a1)
            y1 = CENTER_CIRCLE_RADIUS * math.cos(a1)
            x2 = CENTER_CIRCLE_RADIUS * math.sin(a2)
            y2 = CENTER_CIRCLE_RADIUS * math.cos(a2)
            cv2.line(result, m.to_xy_int((x1, y1, 0)), m.to_xy_int((x2, y2, 0)), color=c, thickness=lw) 

        # Center line
        cv2.line(result, m.to_xy_int((0,YF,0)), m.to_xy_int((0,YN,0)), color=c, thickness=lw)

        return result


def get_pixel_coords(shape):
    H,W = shape
    y = np.arange(0,H)
    x = np.arange(0,W)
    x,y = np.meshgrid(x,y)
    z = [ np.expand_dims(y, axis=0), np.expand_dims(x, axis=0) ]
    z = np.concatenate(z, axis=0)
    return z    

def merge_maps(list_maps):
    return np.max(list_maps, axis=0)


