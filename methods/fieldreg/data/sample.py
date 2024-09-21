from dataclasses import dataclass
import numpy as np


@dataclass
class Sample:
    """ Class for keeping calibration information for one sample 
        position is in meters
        pan, tilt, roll, fov angles are in degrees
    """
    image_name: str
    x: float
    y: float
    z: float
    pan: float
    tilt: float
    roll: float
    fov_v: float
    fov_h: float

    @staticmethod
    def from_calib(c, image_name):
        result = Sample(
            image_name = image_name,
            x = c["position"][0], 
            y = c["position"][1],
            z = c["position"][2],
            pan = c["angles"][0],
            tilt = c["angles"][1],
            roll = c["angles"][2],
            fov_v = c["fov_v"],
            fov_h = c["fov_h"]         
        )
        return result


