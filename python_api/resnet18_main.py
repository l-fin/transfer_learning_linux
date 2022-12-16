from __future__ import print_function, division

import os
import copy
import sys

from PIL import Image

from lfindl_resnet18 import LFINDL_RESNET18

if __name__ == '__main__':

    model_path = sys.argv[1] 
    file_path = sys.argv[2]

    if len(sys.argv) < 3:
        print("Insufficient arguments")
        sys.exit()

    class_names = ['flood/crack', 'non-flood/non-crack']

    resnet18 = LFINDL_RESNET18(model_path)

    image = Image.open(file_path)

    preds = resnet18.predict(image=image)

    print('predicted: {}'.format(class_names[preds]))    