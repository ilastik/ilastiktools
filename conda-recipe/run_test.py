import ilastiktools
import numpy as np 

gridseg = ilastiktools.GridSegmentor_3D_UInt32()

labels = np.arange(27).reshape((3, 3, 3)).astype(np.uint32)
labels //= 10
labels += 1

print('labels', labels.dtype)
feats = np.random.random((3, 3, 3)).astype(np.float32)
print('feats', feats.dtype)

gridseg.preprocessing(labels, feats)
del labels

tmpout = gridseg.serializeGraph()
print(tmpout)
tmp = np.zeros(tmpout.shape, dtype=np.uint32)
tmpout = gridseg.serializeGraph(tmp)
print("same array values?", np.all(tmp == tmpout))
print(tmp)

brushStroke = np.zeros([3,3,3], dtype=np.uint8)
brushStroke[1,1,1] = 1
brushStroke[:,:,2] = 0
brushStroke[:,:,0] = 0
brushStroke[0,:,0] = 0
gridseg.addSeeds(brushStroke=brushStroke,roiBegin=[0,0,0], 
                            roiEnd=[3,3,3], maxValidLabel=2)
gridseg.run(0.5, 0.01)
seg = gridseg.getSegmentation([0,0,0], [3,3,1])
print("Got segmentation of size ", seg.shape)
print(seg)
print('supervoxels', gridseg.getSuperVoxelSeg())
print('SV seeds', gridseg.getSuperVoxelSeeds())
seg = gridseg.getSegmentation([0,0,0], [3,3,1])
seg = gridseg.getSegmentation([0,0,0], [3,3,1])