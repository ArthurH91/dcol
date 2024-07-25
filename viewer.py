from pinocchio import visualize
import meshcat


def create_viewer(rmodel, cmodel, vmodel):
    viz = visualize.MeshcatVisualizer(
        model=rmodel,
        collision_model=cmodel,
        visual_model=vmodel,
    )
    viz.initViewer(viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000"))
    viz.loadViewerModel("pinocchio")

    viz.displayCollisions(True)
    return viz


def add_sphere_to_viewer(viz, sphere_name, radius, position, color=[1, 0, 0, 1]):
    """
    Adds a sphere to the Meshcat visualizer.

    Parameters:
    viz (MeshcatVisualizer): The Meshcat visualizer.
    sphere_name (str): The name of the sphere.
    radius (float): The radius of the sphere.
    position (list or np.array): The position of the sphere as [x, y, z].
    color (list): The color of the sphere as [r, g, b, alpha].
    """
    try:
        # Check if the sphere already exists
        _ = viz.viewer[sphere_name]
        viz.viewer[sphere_name].delete()
    except KeyError:
        # Sphere does not exist
        pass
    sphere_geom = meshcat.geometry.Sphere(radius)
    sphere_material = meshcat.geometry.MeshLambertMaterial(color=color)

    viz.viewer[sphere_name].set_object(sphere_geom, sphere_material)
    viz.viewer[sphere_name].set_transform(
        meshcat.transformations.translation_matrix(position)
    )


def display_closest_points(viz, cp1, cp2, cp1_name: str, cp2_name:str):

    add_sphere_to_viewer(viz, sphere_name=cp1_name, radius=0.05, position=cp1)
    add_sphere_to_viewer(viz, sphere_name=cp2_name, radius=0.05, position=cp2)
