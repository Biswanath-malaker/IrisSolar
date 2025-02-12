import numpy as np
from skimage import measure
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt

# Let's assume I1 and I2 are 2D numpy arrays (same shape)
# Example data:
I1 = np.random.rand(100, 100)
I2 = np.random.rand(100, 100)

delta = 0.15

x = y = np.arange(-3.0, 3.01, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)


Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)

I1 = (Z1-Z2)**2
I2 = I1

# 1. Find the contour(s) in I1. Let's say we use a threshold of 0.5
contours = measure.find_contours(I1, level=0.4)

# 'contours' is a list of (N, 2) arrays, each array is one contour's (row, col) coords.
# If there's more than one contour, pick which one you need; here, we'll just take the first:
contour = contours[0]

# 2. Create a boolean mask for all points *inside* this contour
#    polygon2mask expects (image_shape, polygon), with polygon in (row, col) coords.
mask = polygon2mask(I1.shape, contour)

# 3. Extract pixel values from both images using the mask
vals_I1 = I1[mask]
vals_I2 = I2[mask]


print("Number of pixels in contour region:", np.count_nonzero(mask))
print("Mean intensity in I1:", vals_I1.mean())
print("Mean intensity in I2:", vals_I2.mean())

# (Optional) Visualize the mask
plt.figure()
plt.imshow(I1, cmap='gray')
plt.contour(mask, colors='red', linewidths=1)  # Outline of the filled region
plt.title("Mask overlay on I1")
plt.show()
