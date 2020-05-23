import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Process python training data from CSN.')
parser.add_argument('csnPath', metavar='p0', type=str, nargs='+',
                    help='Path to CodeSearchNet repository.')
parser.add_argument('trainingDataPath', metavar='p1', type=str, nargs='+',
                    help='Path to training data repository.')
args = parser.parse_args()

for i in xrange(0,13):
	inPath = args.csnPath[0] + '/CodeSearchNet/resources/data/python/final/jsonl/train/python_train_' + str(i) + '.jsonl'
	outPath = args.trainingDataPath[0] + '/cs230/gpt-2-csn/src/python_train_' + str(i) + '.txt'
	print('Saving ' + inPath + ' to ' + outPath + '...')
	json = pd.read_json(inPath, lines=True)
	np.savetxt(outPath, json.values, fmt='%s')

print('Complete.')
print('Concatenate the txt files and run the following to preprocess (ETA 1.5hrs): $ python3 encode.py python_train_all.txt python_train_all.npz')
