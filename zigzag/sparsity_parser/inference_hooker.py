import torch as torch
from torchvision.io import read_image
import torchvision
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging


class NetworkInference:
    """
    class for network inference in pytorch
    steps refer to: https://pytorch.org/vision/main/models.html#classification
    """

    def __init__(self,
                 model_name: str = "resnet18",
                 dataset_name: str = "cifar10",
                 img_name: str = "../../../../zigzag_imc_version/sparsity_preprocessing/imagenet/ILSVRC2012_val_00000001.JPEG"):
        """
        parameter initialization
        :param model_name: targeted model, options: [resnet18, resnet50]
        :param dataset_name: targeted dataset name. Load in pkl if == "cifar10"
        :param img_idx: targeted image idx from imagenet, range: 1-40000
        """
        # store input parameters
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model_layer_count = self.get_layer_count_in_model(model_name=model_name)
        # load in cifar10 pkl if needed
        if dataset_name == "cifar10":
            self.dataset = NetworkInference.load_in_cifar10_dataset()
            self.dataset_is_pkl = True
        else:
            self.dataset = None  # not used
            self.dataset_is_pkl = False
        # initialize dict storage for hooked results of intermediate layers
        self.activation_i: dict = {}  # dict for storing input tensors
        self.activation_o: dict = {}  # dict for storing output tensors
        # Step 0: Load in image
        if self.dataset_name == "cifar10":
            self.img = NetworkInference.load_in_cifar10_image(cifar10_dataset=self.dataset, img_idx=1)
        else:
            self.img = read_image(img_name)

        # Step 1: Initialize model with the best available weights
        self.weights = None
        self.model = None
        # update self.weight and self.model
        self.initialize_model(model_name=model_name)
        self.model.eval()

        # Step 2: Initialize the inference transforms
        self.preprocess = self.weights.transforms()

        # Step 3: Apply inference preprocessing transforms
        self.batch = self.preprocess(self.img).unsqueeze(0)

        # Step 4: Integrate the hook to observe results of intermediate layers
        self.inject_hooks(model_name=model_name)

        # Step 5: Use the model and print the predicted category
        self.prediction = self.model(self.batch).squeeze(0).softmax(0)
        self.class_id = self.prediction.argmax().item()
        self.score = self.prediction[self.class_id].item()
        self.category_name = self.weights.meta["categories"][self.class_id]
        # print out the category with the highest score and its score
        logging.info(f"{self.category_name}: {100 * self.score:.1f}%")

    @staticmethod
    def get_layer_count_in_model(model_name: str = "resnet18"):
        """
        get total layer count observed in each model
        :param model_name: targeted model name
        :return layer_count: total layer count
        """
        layer_count_dict: dict = {
            "resnet18": 21,  # last layer inaccurate in energy
            "resnet50": 53,  # 54 in the hooker, but 53 in zigzag onnx
            "vgg19": 19,  # last layer inaccurate in energy
            "mobilenetv3": 54,
            "mobilenetv2": 53,
            "quant_mobilenetv2": 53,
        }
        if model_name in layer_count_dict.keys():
            layer_count = layer_count_dict[model_name]
        else:
            layer_count = -1  # not supported yet
        return layer_count

    def extract_activation_of_an_intermediate_layer(self,
                                                    layer_idx: int = 1,
                                                    img_idx: int = 1,
                                                    img_name: str or None = None,
                                                    return_all_layers: bool = False):
        """
        Extract activation of an intermediate layer
        :param layer_idx: targeted layer idx
        :param img_idx: image index in the pkl dataset. Required when self.dataset_is_pkl == True.
        :param img_name: targeted image file name. Required when self.dataset_is_pkl == False.
        :param return_all_layers: whether return all layers activation
        :return None or activation of the targeted layer, type: np.ndarray
        """
        # Sanity check
        assert 0 <= layer_idx <= self.model_layer_count

        # Flush previous results
        self.activation_i: dict = {}  # dict for storing input tensors
        self.activation_o: dict = {}  # dict for storing output tensors

        # Update targeted image
        if self.dataset_is_pkl:
            self.img = NetworkInference.load_in_cifar10_image(cifar10_dataset=self.dataset, img_idx=img_idx)
        else:
            self.img = read_image(img_name)

        # Apply inference preprocessing transforms
        if self.img.shape[0] != 3:
            logging.warning("Illegal image. Input image does not have 3 RGB channels.")
            return None
        else:
            self.batch = self.preprocess(self.img).unsqueeze(0)

        # Update results of final inference and all layers
        self.prediction = self.model(self.batch).squeeze(0).softmax(0)
        self.class_id = self.prediction.argmax().item()
        self.score = self.prediction[self.class_id].item()
        self.category_name = self.weights.meta["categories"][self.class_id]
        # print out the category with the highest score and its score
        # print(f"input: {img_name}, prediction: {self.category_name}, confidence: {100 * self.score:.1f}%")

        # Fetch inputs of specific layer
        if return_all_layers:
            act: list = [value for value in self.activation_i.values()]
        else:
            if layer_idx == self.model_layer_count:
                act: np.ndarray = self.activation_o[f"layer{layer_idx-1}"]
            else:
                act: np.ndarray = self.activation_i[f"layer{layer_idx}"]

        # Print out verbose information
        # print(f"act shape: {act.shape}")

        return act

    def extract_weight_of_an_intermediate_layer(self,
                                                layer_idx: int = 1,
                                                return_all_layers: bool = False):
        """
        Return weight of an intermediate layer
        """
        weight: list = [v.dequantize().detach().cpu().numpy() for k, v in self.model.state_dict().items() if
                        "weight" in k]
        if return_all_layers:
            pass
        else:
            weight: np.ndarray
            weight = weight[layer_idx]
        return weight

    def initialize_model(self, model_name: str = "resnet18"):
        """
        Initialize weight for networks
        :param model_name: model name, options: [resnet18, resnet50, vgg19]
        :return:
        """
        if model_name == "resnet18":
            self.weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.model = torchvision.models.resnet18(weights=self.weights)
        elif model_name == "resnet50":
            self.weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.model = torchvision.models.resnet50(weights=self.weights)
        elif model_name == "vgg19":
            self.weights = torchvision.models.VGG19_Weights.DEFAULT
            self.model = torchvision.models.vgg19(weights=self.weights)
        elif model_name == "mobilenetv3":  # mobilenet v3 small
            self.weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = torchvision.models.mobilenet_v3_small(weights=self.weights)
        elif model_name == "mobilenetv2":
            self.weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
            self.model = torchvision.models.mobilenet_v2(weights=self.weights)
        elif model_name == "quant_mobilenetv2":  # INT8 quantized model
            self.weights = torchvision.models.quantization.MobileNet_V2_QuantizedWeights.DEFAULT
            self.model = torchvision.models.quantization.mobilenet_v2(weights=self.weights, quantize=True)
        else:
            pass

    def inject_hooks(self, model_name: str = "resnet18"):
        """
        Inject hooks within the model
        :param model_name: targeted model, option: [resnet18, resnet50]
        """
        if model_name == "resnet18":
            self.model.conv1.register_forward_hook(self.get_activation(f'layer0'))
            self.model.layer1[0].conv1.register_forward_hook(self.get_activation(f'layer1'))
            self.model.layer1[0].conv2.register_forward_hook(self.get_activation(f'layer2'))
            self.model.layer1[1].conv1.register_forward_hook(self.get_activation(f'layer3'))
            self.model.layer1[1].conv2.register_forward_hook(self.get_activation(f'layer4'))
            self.model.layer2[0].conv1.register_forward_hook(self.get_activation(f'layer5'))
            self.model.layer2[0].conv2.register_forward_hook(self.get_activation(f'layer6'))
            self.model.layer2[0].downsample[0].register_forward_hook(self.get_activation(f'layer7'))
            self.model.layer2[1].conv1.register_forward_hook(self.get_activation(f'layer8'))
            self.model.layer2[1].conv2.register_forward_hook(self.get_activation(f'layer9'))
            self.model.layer3[0].conv1.register_forward_hook(self.get_activation(f'layer10'))
            self.model.layer3[0].conv2.register_forward_hook(self.get_activation(f'layer11'))
            self.model.layer3[0].downsample[0].register_forward_hook(self.get_activation(f'layer12'))
            self.model.layer3[1].conv1.register_forward_hook(self.get_activation(f'layer13'))
            self.model.layer3[1].conv2.register_forward_hook(self.get_activation(f'layer14'))
            self.model.layer4[0].conv1.register_forward_hook(self.get_activation(f'layer15'))
            self.model.layer4[0].conv2.register_forward_hook(self.get_activation(f'layer16'))
            self.model.layer4[0].downsample[0].register_forward_hook(self.get_activation(f'layer17'))
            self.model.layer4[1].conv1.register_forward_hook(self.get_activation(f'layer18'))
            self.model.layer4[1].conv2.register_forward_hook(self.get_activation(f'layer19'))
            self.model.fc.register_forward_hook(self.get_activation(f'layer20'))
        elif model_name == "resnet50":
            self.model.conv1.register_forward_hook(self.get_activation(f'layer0'))
            self.model.layer1[0].conv1.register_forward_hook(self.get_activation(f'layer1'))
            self.model.layer1[0].downsample[0].register_forward_hook(self.get_activation(f'layer2'))
            self.model.layer1[0].conv2.register_forward_hook(self.get_activation(f'layer3'))
            self.model.layer1[0].conv3.register_forward_hook(self.get_activation(f'layer4'))
            self.model.layer1[1].conv1.register_forward_hook(self.get_activation(f'layer5'))
            self.model.layer1[1].conv2.register_forward_hook(self.get_activation(f'layer6'))
            self.model.layer1[1].conv3.register_forward_hook(self.get_activation(f'layer7'))
            self.model.layer1[2].conv1.register_forward_hook(self.get_activation(f'layer8'))
            self.model.layer1[2].conv2.register_forward_hook(self.get_activation(f'layer9'))
            self.model.layer1[2].conv3.register_forward_hook(self.get_activation(f'layer10'))
            self.model.layer2[0].conv1.register_forward_hook(self.get_activation(f'layer11'))
            self.model.layer2[0].downsample[0].register_forward_hook(self.get_activation(f'layer12'))
            self.model.layer2[0].conv2.register_forward_hook(self.get_activation(f'layer13'))
            self.model.layer2[0].conv3.register_forward_hook(self.get_activation(f'layer14'))
            self.model.layer2[1].conv1.register_forward_hook(self.get_activation(f'layer15'))
            self.model.layer2[1].conv2.register_forward_hook(self.get_activation(f'layer16'))
            self.model.layer2[1].conv3.register_forward_hook(self.get_activation(f'layer17'))
            self.model.layer2[2].conv1.register_forward_hook(self.get_activation(f'layer18'))
            self.model.layer2[2].conv2.register_forward_hook(self.get_activation(f'layer19'))
            self.model.layer2[2].conv3.register_forward_hook(self.get_activation(f'layer20'))
            self.model.layer2[3].conv1.register_forward_hook(self.get_activation(f'layer21'))
            self.model.layer2[3].conv2.register_forward_hook(self.get_activation(f'layer22'))
            self.model.layer2[3].conv3.register_forward_hook(self.get_activation(f'layer23'))
            self.model.layer3[0].conv1.register_forward_hook(self.get_activation(f'layer24'))
            self.model.layer3[0].downsample[0].register_forward_hook(self.get_activation(f'layer25'))
            self.model.layer3[0].conv2.register_forward_hook(self.get_activation(f'layer26'))
            self.model.layer3[0].conv3.register_forward_hook(self.get_activation(f'layer27'))
            self.model.layer3[1].conv1.register_forward_hook(self.get_activation(f'layer28'))
            self.model.layer3[1].conv2.register_forward_hook(self.get_activation(f'layer29'))
            self.model.layer3[1].conv3.register_forward_hook(self.get_activation(f'layer30'))
            self.model.layer3[2].conv1.register_forward_hook(self.get_activation(f'layer31'))
            self.model.layer3[2].conv2.register_forward_hook(self.get_activation(f'layer32'))
            self.model.layer3[2].conv3.register_forward_hook(self.get_activation(f'layer33'))
            self.model.layer3[3].conv1.register_forward_hook(self.get_activation(f'layer34'))
            self.model.layer3[3].conv2.register_forward_hook(self.get_activation(f'layer35'))
            self.model.layer3[3].conv3.register_forward_hook(self.get_activation(f'layer36'))
            self.model.layer3[4].conv1.register_forward_hook(self.get_activation(f'layer37'))
            self.model.layer3[4].conv2.register_forward_hook(self.get_activation(f'layer38'))
            self.model.layer3[4].conv3.register_forward_hook(self.get_activation(f'layer39'))
            self.model.layer3[5].conv1.register_forward_hook(self.get_activation(f'layer40'))
            self.model.layer3[5].conv2.register_forward_hook(self.get_activation(f'layer41'))
            self.model.layer3[5].conv3.register_forward_hook(self.get_activation(f'layer42'))
            self.model.layer4[0].conv1.register_forward_hook(self.get_activation(f'layer43'))
            self.model.layer4[0].downsample[0].register_forward_hook(self.get_activation(f'layer44'))
            self.model.layer4[0].conv2.register_forward_hook(self.get_activation(f'layer45'))
            self.model.layer4[0].conv3.register_forward_hook(self.get_activation(f'layer46'))
            self.model.layer4[1].conv1.register_forward_hook(self.get_activation(f'layer47'))
            self.model.layer4[1].conv2.register_forward_hook(self.get_activation(f'layer48'))
            self.model.layer4[1].conv3.register_forward_hook(self.get_activation(f'layer49'))
            self.model.layer4[2].conv1.register_forward_hook(self.get_activation(f'layer50'))
            self.model.layer4[2].conv2.register_forward_hook(self.get_activation(f'layer51'))
            self.model.layer4[2].conv3.register_forward_hook(self.get_activation(f'layer52'))
            self.model.fc.register_forward_hook(self.get_activation(f'layer53'))  # not exist in onnx model
        elif model_name == "vgg19":
            self.model.features[0].register_forward_hook(self.get_activation(f'layer0'))
            self.model.features[1].register_forward_hook(self.get_activation(f'layer1'))
            self.model.features[5].register_forward_hook(self.get_activation(f'layer2'))
            self.model.features[7].register_forward_hook(self.get_activation(f'layer3'))
            self.model.features[10].register_forward_hook(self.get_activation(f'layer4'))
            self.model.features[12].register_forward_hook(self.get_activation(f'layer5'))
            self.model.features[14].register_forward_hook(self.get_activation(f'layer6'))
            self.model.features[16].register_forward_hook(self.get_activation(f'layer7'))
            self.model.features[19].register_forward_hook(self.get_activation(f'layer8'))
            self.model.features[21].register_forward_hook(self.get_activation(f'layer9'))
            self.model.features[23].register_forward_hook(self.get_activation(f'layer10'))
            self.model.features[25].register_forward_hook(self.get_activation(f'layer11'))
            self.model.features[28].register_forward_hook(self.get_activation(f'layer12'))
            self.model.features[30].register_forward_hook(self.get_activation(f'layer13'))
            self.model.features[32].register_forward_hook(self.get_activation(f'layer14'))
            self.model.features[34].register_forward_hook(self.get_activation(f'layer15'))
            self.model.classifier[0].register_forward_hook(self.get_activation(f'layer16'))
            self.model.classifier[3].register_forward_hook(self.get_activation(f'layer17'))
            self.model.classifier[6].register_forward_hook(self.get_activation(f'layer18'))
        elif model_name == "mobilenetv3":
            self.model.features[0][0].register_forward_hook(self.get_activation(f'layer0'))
            self.model.features[1].block[0][0].register_forward_hook(self.get_activation(f'layer1'))
            self.model.features[1].block[1].fc1.register_forward_hook(self.get_activation(f'layer2'))
            self.model.features[1].block[1].fc2.register_forward_hook(self.get_activation(f'layer3'))
            self.model.features[1].block[2][0].register_forward_hook(self.get_activation(f'layer4'))
            layer_idx = 5
            for i in range(2, 4):
                self.model.features[i].block[0][0].register_forward_hook(self.get_activation(f'layer{layer_idx}'))
                self.model.features[i].block[1][0].register_forward_hook(self.get_activation(f'layer{layer_idx+1}'))
                self.model.features[i].block[2][0].register_forward_hook(self.get_activation(f'layer{layer_idx+2}'))
                layer_idx += 3
            for i in range(4, 12):
                self.model.features[i].block[0][0].register_forward_hook(self.get_activation(f'layer{layer_idx}'))
                self.model.features[i].block[1][0].register_forward_hook(self.get_activation(f'layer{layer_idx+1}'))
                self.model.features[i].block[2].fc1.register_forward_hook(self.get_activation(f'layer{layer_idx+2}'))
                self.model.features[i].block[2].fc2.register_forward_hook(self.get_activation(f'layer{layer_idx+3}'))
                self.model.features[i].block[3][0].register_forward_hook(self.get_activation(f'layer{layer_idx+4}'))
                layer_idx += 5
            self.model.features[12][0].register_forward_hook(self.get_activation(f'layer51'))
            self.model.classifier[0].register_forward_hook(self.get_activation(f'layer52'))
            self.model.classifier[3].register_forward_hook(self.get_activation(f'layer53'))
        elif model_name in ["mobilenetv2", "quant_mobilenetv2"]:
            self.model.features[0][0].register_forward_hook(self.get_activation(f'layer0'))
            self.model.features[1].conv[0][0].register_forward_hook(self.get_activation(f'layer1'))
            self.model.features[1].conv[1].register_forward_hook(self.get_activation(f'layer2'))
            layer_counter = 3
            for i in range(2, 18):
                self.model.features[i].conv[0][0].register_forward_hook(self.get_activation(f'layer{layer_counter}'))
                self.model.features[i].conv[1][0].register_forward_hook(self.get_activation(f'layer{layer_counter+1}'))
                self.model.features[i].conv[2].register_forward_hook(self.get_activation(f'layer{layer_counter+2}'))
                layer_counter += 3
            self.model.features[18][0].register_forward_hook(self.get_activation(f'layer51'))
            self.model.classifier[1].register_forward_hook(self.get_activation(f'layer52'))
        else:
            pass

    def get_activation(self, name):
        """
        Define the hook function to inject within the model
        """
        def hook(model, input, output):
            if len(input) == 1:  # layer has 1 input (such as cnn layer)
                # Case 1: Linear layer with single input
                # print("(Single) Input Shape: ", input[0].shape)
                # print("(Single) Output Shape: ", output.shape)
                self.activation_i[name] = input[0].dequantize().detach().cpu().numpy()
                self.activation_o[name] = output.dequantize().detach().cpu().numpy()
            else:
                pass
        return hook

    @staticmethod
    def convert_imagenet_idx_to_filename(img_idx: int = 1) -> str:
        """
        Convert the imagenet idx to filename from imagenet verification dataset
        :param img_idx: image idx in test dataset, range: 1 - 40000 (40001-50000 is not stored locally)
        :return: file name
        """
        file_idx = "0" * (5 - len(str(img_idx))) + str(img_idx)
        file_name = f"../../../../zigzag_imc_version/sparsity_preprocessing/imagenet/ILSVRC2012_val_000{file_idx}.JPEG"
        return file_name

    @staticmethod
    def derive_density_vector(op_array):
        """
        called within calc_density_distribution: derive the density vector in manually defined dimension ordering
        """
        # old method, but return strange values
        pass

    @staticmethod
    def calc_density_distribution(op_array: np.ndarray, tile_size: int,
                                  enable_relu: bool = False) -> tuple:
        """
        Calculate tile-level sparsity within the given activation according to the tile size
        :param op_array: given operand array
        :param tile_size: targeted tile size
        :param enable_relu: if count negatives as zeros when calculating density
        :return density_list, density_mean, density_std
        """
        # Flatten the array into a 2D array
        assert isinstance(op_array, np.ndarray), \
            "Probably a tensor sit at the input, rather than a ndarray, that does not support the transpose operation."
        ####
        # change flatten ordering, with the bottom always being C, no matter for Weight or Activation
        # [for W]: K, C, FY, FX -> FX, FY, K, C; [for A]: B, C, OY, OX -> OX, OY, B, C
        # (this has tiny impact on distribution std, observed: 10% difference)
        # transposed_array = op_array.transpose((3, 2, 0, 1))
        ####
        # flatten_array = op_array.reshape(op_array.shape[0], -1)
        # Reshape the array corresponding to the tile size
        reshaped_array = op_array.reshape(-1, tile_size)
        # calculate the density of all tiles
        if enable_relu:
            density_vector = np.sum(reshaped_array > 0, axis=1) / reshaped_array.shape[1]
        else:
            density_vector = np.count_nonzero(reshaped_array, axis=1) / reshaped_array.shape[1]
        total_tiles_count = reshaped_array.shape[0]
        assert len(density_vector) == total_tiles_count
        # create tile density list
        tile_density_list: np.ndarray
        tile_density_list = np.array([i/tile_size for i in range(tile_size, -1, -1)])
        # Unique density vector
        density_list: np.ndarray
        counts: np.ndarray
        density_list, counts = np.unique(density_vector, return_counts=True)
        # Note: density_list may not contain all density probability, while tile_density_list contains all
        assert sum(counts) == total_tiles_count
        density_occurrence: np.ndarray = counts / total_tiles_count
        # Insert assertion: 2% approximation mismatch is allowed. This mismatch is due to python division accuracy loss.
        assert 0.98 < sum(density_occurrence) < 1.02
        # allocate tile density occurrence
        tile_density_occurrence: np.ndarray
        if len(density_list) == (tile_size + 1):  # contains all probability
            tile_density_list = density_list
            tile_density_occurrence = density_occurrence
        else:
            # initialize tile_density_occurrence to 0
            tile_density_occurrence = np.zeros(len(tile_density_list), dtype=float)
            for density_idx in range(len(density_list)):
                density_to_find = density_list[density_idx]
                tile_density_idx = np.where(tile_density_list == density_to_find)[0][0]
                tile_density_occurrence[tile_density_idx] = density_occurrence[density_idx]

        # Calculate mean, std
        density_mean: float = round(np.mean(density_vector).item(), 3)
        density_std: float = round(np.std(density_vector).item(), 3)
        # print verbose information
        logging.info(f"density of current image: [mean: {density_mean:<5}, std: {density_std:<5}]")
        return tile_density_list, tile_density_occurrence, density_mean, density_std

    @staticmethod
    def load_in_cifar10_dataset() -> dict:
        """
        return cifar10 dataset in pickle
        """
        # training_batch_filelist = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        test_batch_filelist = ["test_batch"]
        for file in test_batch_filelist:
            with open(f"../../../../zigzag_imc_version/sparsity_preprocessing/cifar-10-batches-py/{file}", "rb") as fo:
                dataset = pickle.load(fo, encoding="bytes")
                return dataset

    @staticmethod
    def load_in_cifar10_image(cifar10_dataset: dict, img_idx: int) -> torch.Tensor:
        """
        fetch cifar10 specific image
        :param cifar10_dataset: cifar10 dataset in dict
        :param img_idx: image index, range: 0-9999
        """
        # return one image object from self.dict, reshape it and convert it to numpy array
        image = cifar10_dataset[b"data"][img_idx]
        # reshape images to C, IY, IX
        image = image.reshape((3, 32, 32))
        # convert to pytorch tensor
        image = torch.from_numpy(image)
        return image


if __name__ == "__main__":
    """
    Debug code for density (not sparsity) extraction from network on imagenet dataset
    :param dataset_name: dataset name. options: [cifar10, imagenet]
    :param model_name: model name. options: [resnet18, resnet50, vgg19, mobilenetv2, mobilenetv3, quant_mobilenetv2]
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    ###############################################################################
    dataset_name = "imagenet"
    model_name = "resnet18"
    nn = NetworkInference(model_name=model_name, dataset_name=dataset_name)
    # img_indices = np.random.randint(1, 40000, size=100)
    img_indices = [1]
    ###############################################################################
    for img_idx in img_indices:
        logging.info(f"image index: {img_idx}")
        if dataset_name == "cifar10":
            img_name = None
        else:
            img_name = NetworkInference.convert_imagenet_idx_to_filename(img_idx=img_idx)
        intermediate_act = nn.extract_activation_of_an_intermediate_layer(layer_idx=1,
                                                                          img_idx=img_idx,
                                                                          img_name=img_name)
        ans = NetworkInference.calc_density_distribution(activation=intermediate_act, tile_size=8)
