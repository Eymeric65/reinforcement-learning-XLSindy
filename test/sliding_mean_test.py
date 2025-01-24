import numpy as np
from rl_util.environment import SlidingMeanStandard


data_shape = (10,4)
batch_axis = None

data_mean = 3
data_std = 8

iteration = 1000

DataSlidingMeanStd = SlidingMeanStandard(
    batch_axis=batch_axis,
    moving_proportion=0.1
)

def DebugSliding(batch,total):

    DataSlidingMeanStd.update(batch)

    total = np.concat((batch,total),axis=batch_axis)

    var = np.var( total,axis=batch_axis)
    mean = np.mean( total,axis=batch_axis)

    print("Class estimate :",DataSlidingMeanStd.mean,DataSlidingMeanStd.var**0.5)
    print("Numpy estimate :",mean,var**0.5)
    return total


sample = np.random.normal(size=data_shape,loc=data_mean,scale=data_std)

DataSlidingMeanStd.update(sample)

print("0 Class estimate :",DataSlidingMeanStd.mean,DataSlidingMeanStd.var**0.5)
print("0 Numpy estimate :",np.mean( sample,axis=batch_axis),np.var( sample,axis=batch_axis)**0.5)

for i in range(iteration):

    batch = np.random.normal(size=data_shape,loc=data_mean,scale=data_std)

    print("the batch size : ",batch.shape)

    sample =DebugSliding( batch , sample)

    print("the shape of the total sample is :",sample.shape)

    sample_normalised = DataSlidingMeanStd.normalise(sample)

    print("normalised sample :",np.mean( sample_normalised,axis=batch_axis),np.var( sample_normalised,axis=batch_axis)**0.5)
