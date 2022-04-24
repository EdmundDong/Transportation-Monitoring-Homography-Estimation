# Transportation-Monitoring-Homography-Estimation

## Background Extraction

## HomographyNet
The main purpose of the HomographyNet submodule is to estimate the relative homography between a pair of images. In practical terms, this estimation would be the matrix transformation that would warp one image to appear to be from the same perspective as another. In the context of this project, the goal of the submodule is to input a camera view of an intersection and a satellite view of that same intersection. 

### Submodule Workflow

1. Input images must be placed in a named folder under `HomographyNet/data/<folder>`. The name of this folder must be updated in config.py on line 12 as `image_folder = 'data/<folder>'` Input files must contain specific suffixes to work correctly. Images to be compared should be named `<name>-f1.jpg` and `<name>-f2.jpg` for each corresponding pair. The ground truth matrix should be in a file named `<name>-m.csv`. The ground truth matrix should contain the 4-point representation of image `f1` in the first 4 lines followed by the 4-point representation of image `f2` in the next 4 lines. Examples of a dataset can be found in the HomographyNet submodule under `HomographyNet\data\dataset1`. Images are randomized, this can be disabled by removing line 140 in `preprocess.py` (`np.random.shuffle(files)`)

2. Images must be pre-processed into pickles for training and testing with `pre_process.py`. The current implementation will split the input files into a training, validating, and testing set. As images are compiled for pickling, the network input and targets are placed in `HomographyNet\output\preprocess` for each image that is pickled. 

3. Training is completed with `train.py`. Valid arguments for training are `--end-epoch`, `--lr` (to set the starting learning rate), `--momentum`, `--batch-size`, `--checkpoint` (to resume training). Checkpoints are automatically saved during training. Tensorboard can be opened to view training statistics using `tensorboard --logdir runs`. 

4. Testing is completed with `test.py`. As images are tested, the network output and targets are placed in `HomographyNet\output\test` for each image pair that is tested. Testing can also be done with traditional homography estimation algorithms (SURF + RANSAC, Identity Homography) using `test_orb.py`.

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
