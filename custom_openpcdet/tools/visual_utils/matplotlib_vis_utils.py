import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_scenes(points, ref_boxes=None, ref_scores=None, ref_labels=None, **kwargs):
    if hasattr(points, 'cpu'): points = points.cpu().numpy()
    if ref_boxes is not None and hasattr(ref_boxes, 'cpu'): ref_boxes = ref_boxes.cpu().numpy()
    if ref_scores is not None and hasattr(ref_scores, 'cpu'): ref_scores = ref_scores.cpu().numpy()
    
    # Dark Mode 3D aesthetics
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample points for stable 3D performance in Matplotlib
    skip = 5
    p_sub = points[::skip]
    ax.scatter(p_sub[:, 0], p_sub[:, 1], p_sub[:, 2], s=0.1, c=p_sub[:, 2], cmap='viridis', alpha=0.5)
    
    if ref_boxes is not None:
        for i in range(len(ref_boxes)):
            x, y, z, l, w, h, heading = ref_boxes[i]
            
            # Corner generation for 3D Bounding Boxes
            dx, dy, dz = l/2, w/2, h/2
            corners = np.array([
                [dx, dy, -dz], [-dx, dy, -dz], [-dx, -dy, -dz], [dx, -dy, -dz],
                [dx, dy, dz], [-dx, dy, dz], [-dx, -dy, dz], [dx, -dy, dz]
            ])
            cosa, sina = np.cos(-heading), np.sin(-heading)
            rot_mat = np.array([
                [cosa, -sina, 0],
                [sina, cosa, 0],
                [0, 0, 1]
            ])
            corners = corners.dot(rot_mat.T)
            corners[:, 0] += x
            corners[:, 1] += y
            corners[:, 2] += z
            
            # Draw wireframe edges
            edges = [
                [0,1], [1,2], [2,3], [3,0],
                [4,5], [5,6], [6,7], [7,4],
                [0,4], [1,5], [2,6], [3,7]
            ]
            for edge in edges:
                ax.plot([corners[edge[0],0], corners[edge[1],0]], 
                        [corners[edge[0],1], corners[edge[1],1]], 
                        [corners[edge[0],2], corners[edge[1],2]], color='lime', linewidth=1.5)
            
            # Add confidence score above the box
            if ref_scores is not None:
                ax.text(x, y, z + h + 0.5, f'{ref_scores[i]:.2f}', color='red', fontsize=10, weight='bold', ha='center')
            
    # Set 3D bounds corresponding roughly to frontal point cloud range
    ax.set_xlim(0, 52)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-3, 3)
    ax.set_title("Interactive 3D Lidar Predictions", fontsize=14, color='white')
    ax.axis('off')
    
    # Open fully interactive native Matplotlib window
    plt.show()
