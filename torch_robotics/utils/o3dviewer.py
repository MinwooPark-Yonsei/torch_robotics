import numpy as np
import open3d as o3d

class PointcloudVisualizer():
    def __init__(self) -> None:
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

    def add_geometry(self, cloud):
        self.vis.add_geometry(cloud)

    def update(self, cloud):
        # Update point cloud
        self.vis.update_geometry(cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

        # View control: adjust zoom, front, lookat, and up direction
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.05)
        view_control.set_front([0.0, 0.0, -1.0])
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_up([0.0, -1.0, 0.0])

    def close(self):
        self.vis.destroy_window()

if __name__ == "__main__":
    visualizer = PointcloudVisualizer()
    cloud = o3d.io.read_point_cloud("../../../assets/dataset/one_door_cabinet/46145_link_0/point_sample/full_PC.ply")
    
    # Adding the point cloud geometry
    visualizer.add_geometry(cloud)
    
    try:
        while True:
            # Perform updates and refresh the visualization window
            visualizer.update(cloud)
            xyz = np.asarray(cloud.points)
            xyz *= 1.001  # Example transformation for points update
    except KeyboardInterrupt:
        # Graceful shutdown when the user interrupts
        print("Closing visualization")
        visualizer.close()