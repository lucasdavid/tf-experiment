import numpy as np


def bboxes_to_segmentation_label(labels, bboxes, shape):
  segs = np.zeros(shape, dtype=np.float32)
  H, W = shape[2:]
  for i in range(len(labels)):
    for l, b in zip(labels[i], bboxes[i]):
      bi = (b*(H, W, H, W)).astype('int')
      ymin, xmin, ymax, xmax = bi
      segs[i, l, ymin:ymax, xmin:xmax] = 1
  return segs
