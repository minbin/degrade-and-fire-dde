import unittest
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import join

from pyESN import ESN

def test(fn, tl, esn):
  print('testing '+fn+'...')
  data = []
  with open(fn) as f:
    for l in f:
      data.append(float(l))

  data = np.array(data[:tl])
  data = np.apply_along_axis(np.log, 0, data)

  trainlen = tl
  future = tl * 4

  print('fitting...')
  pred_training = esn.fit(np.ones(trainlen),data[:trainlen])
  print('predicting...')
  prediction = esn.predict(np.ones(future), continuation=True)
  data = np.apply_along_axis(np.exp, 0, data)
  prediction = np.apply_along_axis(np.exp, 0, prediction)
  print('plotting...')
  plt.figure(figsize=(50,20))
  plt.plot(range(0, trainlen), data[0:trainlen], 'b', label="actual")
  plt.plot(range(trainlen+1, trainlen+future+1), prediction, 'r', label="prediction")
  figname = fn[:-4]+'_predict.png'
  print('saving '+figname)
  plt.savefig(figname)

esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = 5000,
          sparsity = 0.95,
          spectral_radius = 2.5)

test(join('prob4', '300_80_10_10_0_1_(0,100).txt'), 1200, esn)

#print("test error: \n"+str(np.sqrt(np.mean((prediction.flatten() - data[trainlen:trainlen+future])**2))))

#plt.plot(range(0,trainlen+future),data[0:trainlen+future],'k',label="target system")
#plt.plot(range(trainlen,trainlen+future),prediction,'r', label="free running ESN")
#lo,hi = plt.ylim()
#plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')

#plt.legend(loc=(0.61,1.1),fontsize='x-small')
