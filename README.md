# Transportation-Monitoring-Homography-Estimation

## Background Extraction
The purpose of the Background Extraction submodule is to enhance homography transformation results through the removal of objects that may alter the output. In this case, we consider such objects to be vehicles.

### Submodule Workflow

1. Input images can be placed anywhere, as long as the folder is accessible, and the proper path is provided at the command line. Any output directory can be specified, if the path does not exist a folder will be automatically created.
2. Running inference is as simple as `python test.py --input [PATH_TO_INPUT_DIR] --output [PATH_TO_OUTPUT_DIR]`, all image processing is handled internally. Output images will have the same file name and resolution as the input images. If the `--input` and `--output` command line arguments are the same, images in the `--input` directory will be saved over.
3. `train_segnet.ipynb` is provided for the purpose of training the semantic segmentation stage. To train the inpainting model, please refer to the submodule README

### Relevant Folder and File Descriptions
| Folder/File | Descriptions |
| -- | -- |
| scripts/ | Script to generate files listing files to be used for training, validation, and testing |
| checkpoints/ | Folder containing semantic segmentation and inpainting weights |
| src/ | This folder contains the bulk of the functionality |
| config.py | training config for image size, batch size, input folder, and training parameters |
| main.py | Creates necessary models and begins inference |
| test.py | Starts `main.py` with inference mode |
| requirements.txt | Contains a list of necessary dependencies to install |
| train_segnet.ipynb | Contains code for purpose of training SegNet |
| dataset.py | Stores images, produced edges, and produced masks required for inference |
| edge_connect.py | Runs inference on images |
| generate_dataset.py | Various collected functions used to generate the datasets in the proper formats required for EdgeConnect training |
| image_proc.py | Handles image slicing, image stitching, and cleanup of folders used for those purposes |
| loss.py | Implements various loss functions |
| models.py | Contains propogation methods for the edge and inpainting models |
| networks.py | Contains model architectures |
| segmentor.py | Uses semantic segmentation to remove vehicles and generate corresponding masks |
| utils.py | Various utility functions used throughout the project |


## HomographyNet
The main purpose of the HomographyNet submodule is to estimate the relative homography between a pair of images. In practical terms, this estimation would be the matrix transformation that would warp one image to appear to be from the same perspective as another. In the context of this project, the goal of the submodule is to input a camera view of an intersection and a satellite view of that same intersection. 

### Submodule Workflow

1. Input images must be placed in a named folder under `HomographyNet/data/<folder>`. The name of this folder must be updated in config.py on line 12 as `image_folder = 'data/<folder>'` Input files must contain specific suffixes to work correctly. Images to be compared should be named `<name>-f1.jpg` and `<name>-f2.jpg` for each corresponding pair. The ground truth matrix should be in a file named `<name>-m.csv`. The ground truth matrix should contain the 4-point representation of image `f1` in the first 4 lines followed by the 4-point representation of image `f2` in the next 4 lines. Examples of a dataset can be found in the HomographyNet submodule under `HomographyNet\data\dataset1`. Images are randomized, this can be disabled by removing line 140 in `preprocess.py` (`np.random.shuffle(files)`)

2. Images must be pre-processed into pickles for training and testing with `pre_process.py`. The current implementation will split the input files into a training, validating, and testing set. As images are compiled for pickling, the network input and targets are placed in `HomographyNet\output\preprocess` for each image that is pickled. 

3. Training is completed with `train.py`. Valid arguments for training are `--end-epoch`, `--lr` (to set the starting learning rate), `--momentum`, `--batch-size`, `--checkpoint` (to resume training). Checkpoints are automatically saved during training (`BEST_checkpoint.tar`, `BEST_model.pt`, `checkpoint.tar`, `model.pt`). Tensorboard can be opened to view training statistics using `tensorboard --logdir runs`. 

4. Testing is completed with `test.py`. As images are tested, the network output and targets are placed in `HomographyNet\output\test` for each image pair that is tested. The files `BEST_model.pt` is used for testing. Testing can also be done with traditional homography estimation algorithms (SURF + RANSAC, Identity Homography) using `test_orb.py`.

### Relevant Folder and File Descriptions
| Folder/File | Descriptions |
| -- | -- |
| cuda/ | script to test if CUDA is working with PyTorch |
| data/ | input images and pickle files |
| models/ | manually saved backups of models from training |
| output/ | outputs from `preprocess.py` and `test.py` |
| runs/ | tensorboard run information |
| config.py | training config for image size, batch size, input folder, and training parameters |
| data_gen.py | dataset handler for training and testing |
| mobilenet_v2.py | implementation of MobileNetV2 |
| pre_process.py | pickles dataset files into training, validating, and testing sets |
| test.py | infer homography of images in test set using HomgraphyNet |
| test_orb.py | infer homography of images in test set using SURF + RANSAC or Identity Homography |
| train.py | train network |
| utils.py | handler for general utilities like saving checkpoints or parsing training arguments |

### Notes

- Estimates are done in 4-point correspondence and therefore training is resolution dependent. 
- The current implementation of this project loads the entire working dataset into RAM for training and inference. The amount of RAM needed scales with dataset size and desired input resolution for training or inference. 
- Implementation has only been tested on a CUDA system. 

## IPM Transformation
