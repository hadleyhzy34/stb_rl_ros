from os import read
import numpy as np
import pdb
import pickle


if __name__ == "__main__":
    pdb.set_trace()
    # data = [1.,1.,1.]
    # file = open('data.pkl','ab')
    # pickle.dump(data,file)
    # pickle.dump(data,file)
    # file.close()
    #
    # read_file = open('data.pkl','rb')
    # data = pickle.load(read_file)
    arr = np.loadtxt("checkpoint/dataset_0.csv", delimiter=',',dtype=float)
    print(arr)
