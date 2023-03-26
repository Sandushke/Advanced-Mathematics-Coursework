# Question 4
# a)
import numpy as np
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
import scipy.fftpack as sfft

image = mpimg.imread("Fruit.jpg")

#fft
imagef = sfft.fft2(image)
plt.imshow(np.abs(imagef))
#image with fft shift
imagef = sfft.fftshift(imagef)
plt.imshow(np.abs(imagef))

#remove high frequencies
Image = np.zeros((360, 360), dtype=complex)
c = 180
r = 50
for m in range(0, 360):
    for n in range(0, 360):
        if np.sqrt(((m-c)**2 + (n-c)**2))<r:
            Image[m, n] = imagef[m, n]

plt.imshow(np.abs(Image))
image1 = sfft.ifft2(Image)
plt.imshow(np.abs(image1))
# plt.show()

#remove low frequencies
Imagef1 = np.zeros((360, 360), dtype=complex)
c = 180
r = 90

for m in range(0, 360):
    for n in range(0, 360):
        if np.sqrt(((m-c)**2 + (n-c)**2))>r:
            Imagef1[m, n] = imagef[m, n]

plt.imshow(np.abs(Imagef1))
image1 = sfft.ifft2(Imagef1)
plt.imshow(np.abs(image1))
plt.show()

# b)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.fftpack as sfft
import scipy.signal as signal

#load the image
img = mpimg.imread("Fruit.jpg")

#Gaussian filter
image = np.outer(signal.gaussian(360, 5), signal.gaussian(360, 5))
imagef = sfft.fft2(sfft.ifftshift(image))  #freq domain kernel
plt.imshow(np.abs(imagef))

imgf = sfft.fft2(img)
plt.imshow(np.abs(imagef))

imgB = imgf * imagef
plt.imshow(np.abs(imgB))

img1 = sfft.ifft2(imgB)
plt.imshow(np.abs(img1))
plt.show()

# c)
import numpy as np
from scipy.fftpack import dct
from PIL import Image

# Load the image
image = Image.open('Fruit.jpg')

# Convert the image to a numpy array
img = np.array(image)

#DCT to the fruit image
DCT = dct(img)

#Scale the image
resizedImage = image.resize((240, 240))

#Plot the original vs compressed fruit image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original fruit image')
plt.subplot(1, 2, 2)
plt.imshow(resizedImage, cmap='gray')
plt.title('Resized fruit image')
plt.show()

# d)
from PIL import Image

# Open the image
image = Image.open('Fruit.jpg')

# Set the image quality to 50 pixels
image.save('compressedImage.jpg', "JPEG", quality=20)

compressedImage = Image.open("compressedImage.jpg")

#Plot the original vs compressed fruit image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original fruit image')
plt.subplot(1, 2, 2)
plt.imshow(compressedImage, cmap='gray')
plt.title('Compressed fruit image')
plt.show()