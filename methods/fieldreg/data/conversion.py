
import cv2
import numpy as np
import methods.fieldreg.data.playfield as playfield


def _yards(x):
    return x * 1.0936132983
    
    
class HomographyToDistanceMap:
    def __init__(
        self,
        playfield_grid_shape=(8,7),
        distance_map_radius=4,
        distance_map_shape=(3840, 2160)
    ):
        self.pf = playfield.Playfield(grid_shape=playfield_grid_shape)
        self.distance_map_radius = distance_map_radius
        self.distance_map_shape = distance_map_shape
        
        # Prepare distance maps
        dms = self.pf.get_distance_map(
            image_shape=distance_map_shape,
            radius=distance_map_radius
        )
        channels = [ np.expand_dims(i, axis=2) for i in dms ]
        channels = np.concatenate(channels, axis=2)
        self.distance_maps = channels
        
        # Prepare scale matrix - Evil WC2014 homography is in yards !
        self.scale = np.array([
            [ _yards(self.pf.length) / distance_map_shape[1], 0.0, 0.0],
            [ 0, _yards(self.pf.width) / distance_map_shape[0], 0.0],
            [ 0, 0, 1]
        ])
                

    def to_distance_map(self, inv_homo, output_shape):
        """ Returns distance map as seen under the given homography """    
                
        # Warp under the given homography
        dmaps = cv2.warpPerspective(
            self.distance_maps, 
            inv_homo, 
            (output_shape[1], output_shape[0]),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0)
        )
        return dmaps
        
    
        
    
    
    
    