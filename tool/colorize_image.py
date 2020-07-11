from PIL import Image
import os
import numpy as np
import cv2
def colorize():
  images_path = '/content/drive/My Drive/Segmentation/voc2012/test_random'
  labels_path = '/content/drive/My Drive/Segmentation/voc2012/result/test/test_random/color' 
  save_path = '/content/drive/My Drive/Segmentation/voc2012/result/test/test_random/colorOverlay'
  fileNames = [fname for fname in os.listdir(labels_path) if fname[-4:]=='.png']
  imagesfileNames = [fname for fname in os.listdir(images_path)]

  try:
    os.mkdir(save_path)
  except:
    pass
  for imageFileName in imagesfileNames:
    imageName = os.path.splitext(imageFileName)[0]
    if imageName + '.png' in fileNames:
      imagePath = os.path.join(images_path, imageFileName)
      labelPath = os.path.join(labels_path, imageName + '.png')
      image = cv2.imread(imagePath)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      label = cv2.imread(labelPath)
      label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

      out_image = image * 0.5 + label * 0.5
      pil_img = Image.fromarray(out_image.astype(np.uint8))
      pil_img.save(os.path.join(save_path, imageFileName))
