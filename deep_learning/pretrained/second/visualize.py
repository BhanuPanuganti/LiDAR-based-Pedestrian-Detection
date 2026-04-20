import open3d as o3d
import numpy as np

def visualize_lidar_with_boxes(bin_path, pred_boxes):

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.paint_uniform_color([0.2, 0.2, 0.2])

    elements = [pcd]

    for box in pred_boxes:
        center = box[0:3]
        dims = box[3:6]
        heading = box[6]

        rot = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, heading))
        obb = o3d.geometry.OrientedBoundingBox(center, rot, dims)
        obb.color = (1, 0, 0)

        elements.append(obb)

    o3d.visualization.draw_geometries(elements)