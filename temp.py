from PIL import Image

from utils import center_crop_resize

im = Image.open("test.tif")
im_out = center_crop_resize(im, (224, 224))
im_out.show()
im_out.save("test_out.tif")
