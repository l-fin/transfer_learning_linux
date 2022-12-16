from __future__ import print_function, division

import os
import copy
import sys

from PIL import Image

from lfindl_mobilenetv3 import LFINDL_MOBILENETV3

if __name__ == '__main__':

    model_path = sys.argv[1] 
    file_path = sys.argv[2]

    if len(sys.argv) < 3:
        print("Insufficient arguments")
        sys.exit()

    class_names = ['flood/crack', 'non-flood/non-crack']

    mobilenetv3 = LFINDL_MOBILENETV3(model_path)

    image = Image.open(file_path)

    preds = mobilenetv3.predict(image=image)

    print('predicted: {}'.format(class_names[preds]))