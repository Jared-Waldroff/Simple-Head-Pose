import posemodels


def load(model_name, mesh_type="mp", mesh_conf=0.5, mesh_iou=0.5):
    """
    Load and return a model based on the given model name.

    Parameters:
    - model_name (str): Name of the model to load (e.g., "svr", "xgboost").
    - mesh_type (str): Type of mesh to use (default is "mp" for MediaPipe).
    - mesh_conf (float): Confidence threshold for the mesh model (default is 0.5).
    - mesh_iou (float): IoU threshold for the mesh model (default is 0.5).

    Returns:
    - A model instance from the posemodels library.
    """

    # Dictionary mapping model names to their respective classes in posemodels
    model_map = {
        "svr": posemodels.Regressor,
        "xgboost": posemodels.Regressor
        # Add other models here as needed
    }

    # Check if the model name is supported
    if model_name in model_map:
        return model_map[model_name](model_name, mesh_type, mesh_conf, mesh_iou)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Available models are: {list(model_map.keys())}")

# Usage Example:
# model = load("svr", mesh_type="mp", mesh_conf=0.25, mesh_iou=0.45)
