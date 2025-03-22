<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:465aff76114bb6ab590e80f635d2f47a104c6ce38af2bb8c49b98cf43dd83d2e
size 1136
=======
# TNODiscoverer
This repository provides tools to find TNOs with deep learning.

### [Workflow]  

Steps 1-3 are for training the model, and steps 4-6 are for using the model to detect TNOs.

|Step|File|Input|Output|Purpose|
|-|-|-|-|-|
|1|ImageCutter.ipynb|.fits (with artificial moving objects), .plantlist (artificial objects info)| .npy|Extract sub-images for training|
|2|Concatenator.ipynb|.npy (sub-images from ImageCutter)|.npy|Prepare dataset for training|
|3|Trainer.ipynb|.npy (dataset from Concatenator), .npy (target information)|.h5 (trained CNN models)|Train the model|
|-|-|-|-|-|
|4|ImageCutter.ipynb|.fits (without artificial moving objects)|.npy|Extract sub-images for detection|
|5|Predictor.ipynb|.npy (sub-images from ImageCutter), .npy (target info), .h5 (model)|.npy|Apply trained model to detect objects|
|6a|Link_sources_to_objects.py|.npy (classification and regression output from Predictor)|.npy|Detect moving objects (linear fitting method)|
|6b|CandidateFinder.ipynb|.npy (classification output from Predictor), .npy (sub-images, target info)|.csv|Detect moving objects (scoring method)|

### Example Files
Due to the repository's capacity limit, only example files are included:  
- Only 4 FITS files out of 44 × 36 total.  
- Only 44 plant list files out of 44 × 36 total.  
- Only 1 sub-image dataset from 1 CCD out of 36 CCDs available.  
- Only 2 models (MobileNet classification and regression).  
- Predicted values for sub-images of the first CCD.  
>>>>>>> cb1b983fa934ee32496cd493906b7eb684d9ba58
