from imgaug import augmenters as iaa
import imgaug as ia

rand_aug: iaa.RandAugment = None

from .default import Default


class RandAug(Default):
  def __init__(
      self,
      n: int = 3,
      m: int = 7,
      seed: int = 10482,
  ):
    self.rand_aug = iaa.RandAugment(n=n, m=m)

    ia.seed(seed)

  def augment(self, image):
    image = self.rand_aug(images=image.numpy())

    return image
