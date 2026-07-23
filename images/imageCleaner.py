import cv2
import numpy as np

# Load image
img = cv2.imread('/home/viper/Documents/GitHub/CelestialNavigation/images/PixelImage1TEST.jpg')

if img is None:
    print("Image not found! Check the path.")
else:
    print("Image loaded successfully:", img.shape)
# --- Method 1: Gaussian Blur (fast, simple) ---
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# --- Method 2: Median Blur (great for salt-and-pepper noise) ---
median = cv2.medianBlur(img, 5)

# --- Method 3: Non-Local Means (best quality, slower) ---
# For color images:
nlm = cv2.fastNlMeansDenoisingColored(img, None,
    h=10,           # filter strength (higher = more denoising, less detail)
    hColor=10,      # same but for color
    templateWindowSize=7,
    searchWindowSize=21
)
# For grayscale images, use: cv2.fastNlMeansDenoising(img, ...)

# --- Method 4: Bilateral Filter (smooths noise, preserves edges) ---
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Save results
cv2.imwrite('gaussian.jpg', gaussian)
cv2.imwrite('median.jpg', median)
cv2.imwrite('nlm.jpg', nlm)
cv2.imwrite('bilateral.jpg', bilateral)

