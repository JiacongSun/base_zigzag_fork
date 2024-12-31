import pickle


def save_to_pickle(obj, filename):
    with open(filename, "wb") as fp:
        pickle.dump(obj, fp)


def read_pickle(filename):
    with open(filename, "rb") as fp:
        load = pickle.load(fp)
    return load


def derive_model_layer_count(model_name: str):
    # Class NetworkInference in inference_hooker.py has its own function to derive this info
    layer_count_dict: dict = {
        "resnet18": 21,
        "resnet18_sparse": 17,  # does not have downsample layers (4) and fc layer (1)
        "resnet50": 54,  # 54 in the hooker, but 53 in zigzag onnx
        "vgg19": 19,
        "mobilenetv3": 54,
        "mobilenetv2": 53,
        "mobilenetv2_sparse": 50,  # see comments of resnet18_sparse
        "quant_mobilenetv2": 53,
        "efficientnetb0_sparse": 49,  # see comments of resnet18_sparse
    }
    return layer_count_dict[model_name]