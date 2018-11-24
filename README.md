# Unet Implementation is Keras with GPU

## How to use
```
python3 train.py [-h] -d DATA [-s SIZE] [-c CLASSES] [-sg SKIP_GRIDDING]
```

See description below to see all available options
optional arguments:
```
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Input directory containing images and labels
  -s SIZE, --size SIZE  Input size of image to be used. [Default] = 200
  -c CLASSES, --classes CLASSES
                        Input number of classes.[Default] = 1
  -sg SKIP_GRIDDING, --skip_gridding SKIP_GRIDDING
                        If gridding is already done then skip it. [Default] is
                        No = 0
```
