from __future__ import print_function
import sys
import os
import os.path
import csv
import re
import random
import datetime
import numpy as np
import pandas as pd
import copy
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)



def main(argv):

    a = torch.ones(5)
    b = a.numpy()

    a.add_(1)

    print(a)
    print(b)

    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)

    x = b
    y = x.add_(1)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        z = x + y
        
        print(x)
        print(y)
        print(z)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_deviceS"] = "0,1,2,3"
    main(sys.argv)