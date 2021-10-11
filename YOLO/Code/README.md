# Yolo v1 코드 구현

## Train

    python train.py

### Hyperparameters etc.
    LEARNING_RATE = 2e-5
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    WEIGHT_DECAY = 0
    EPOCHS = 1000
    NUM_WORKERS = 2
    PIN_MEMORY = True
    LOAD_MODEL = False
    LOAD_MODEL_FILE = "overfit.pth.tar"
    IMG_DIR = "data/images"
    LABEL_DIR = "data/labels"


## reference
    https://www.youtube.com/watch?v=n9_XyCGr-MI
    https://github.com/aladdinpersson/Machine-Learning-Collection