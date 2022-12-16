from __future__ import print_function, division

import os
import copy
import sys

from PIL import Image

from lfindl_resnet50 import LFINDL_RESNET50

if __name__ == '__main__':

    model_path = sys.argv[1] 
    file_path = sys.argv[2]

    if len(sys.argv) < 3:
        print("Insufficient arguments")
        sys.exit()

    class_names = ['flood/crack', 'non-flood/non-crack']

    resnet50 = LFINDL_RESNET50(model_path)

    image = Image.open(file_path)

    preds = resnet50.predict(image=image)

    print('predicted: {}'.format(class_names[preds]))    