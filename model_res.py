def run_model(model):
    import pandas as pd
    from numpy import asarray
    from numpy import savetxt
    from pandas import read_csv
    import json
    import sys

    import numpy as np
    from numpy import genfromtxt

    name_of_actions = ["clockwise-rotation", "counterclockwise-rotation", "eight",
                       "flip", "horizontal-rotation", "horizontal-x-shake", "horizontal-y-shake",
                       "infinity", "vertical-x-shake", "vertical-y-shake", "hold"]

    data_folder = "./data/data1.csv"

    data_seq = np.array([genfromtxt(data_folder, delimiter=',')])
    print(data_seq.shape, file=sys.stderr)

    return_val = model.predict(data_seq)

    max1, max2, max3 = -1, -1, -1
    ind = 0
    for pos in return_val[0]:
        if max1 == -1 or pos > return_val[0][max1]:
            if max1 != -1 and(return_val[0][max1] > return_val[0][max2] or max2 == -1):
                if max2 != -1 and (max3 == -1 or return_val[0][max2] > return_val[0][max3]):
                    max3 = max2
                max2 = max1
            max1 = ind
        ind += 1

    answer1 = {
      "label": str(name_of_actions[max1]),
      "prob": str(return_val[0][max1]),
    }
    answer2 = {
      "label": str(name_of_actions[max2]),
      "prob": str(return_val[0][max2]),
    }
    answer3 = {
      "label": str(name_of_actions[max3]),
      "prob": str(return_val[0][max3]),
    }
    answer = json.dumps([answer1, answer2, answer3])
    return answer
