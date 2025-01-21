import pickle
import logging
from api import read_pickle, derive_model_layer_count


def show_layer_shape(model_name: str = "resnet18", layer_id: int = 2):
    pkl_layer_shape = f"./pkl/layer_shape/{model_name}/shape_{model_name}.pkl"
    with open(pkl_layer_shape, "rb") as fp:
        workload = pickle.load(fp)[layer_id]
    return workload


if __name__ == "__main__":
    """
    reporting layer shape
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    ##############################
    ## parameters
    model_name = "resnet50"
    ##############################
    layer_count = derive_model_layer_count(model_name)
    for layer_id in range(layer_count):
        workload = show_layer_shape(model_name=model_name, layer_id=layer_id)
        logging.info(f"layer {layer_id}: {workload}")
