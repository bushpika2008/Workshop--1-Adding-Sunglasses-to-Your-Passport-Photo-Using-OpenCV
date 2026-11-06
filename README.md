# Workshop--1-Adding-Sunglasses-to-Your-Passport-Photo-Using-OpenCV
# Name : BUSHPIKA C
# register no  :212225230038

## Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

### Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.
### Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations
### How to Use:
- Clone this repository.
- Add your passport-sized photo to the images folder.
- Run the script to see your "cool" transformation!
### Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows

```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

```
# Load the Face Image
faceImage = cv2.imread('bushpassport.jpeg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
```
```
faceImage.shape
```

```
#resized_faceImage.shape
faceImage.shape
```
```
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('Sunglass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
```
```
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(250, 80))
print("image Dimension ={}".format(glassPNG.shape))
```
```
# Separate color and alpha channels
glassBGR = glassPNG[:, :, :3]
glassMask1 = glassPNG[:, :, 3]
```
```
# Display sunglasses and alpha mask
plt.figure(figsize=(12,5))

plt.subplot(121)
plt.imshow(glassBGR[:,:,::-1])
plt.title("Sunglasses")

plt.subplot(122)
plt.imshow(glassMask1, cmap="gray")
plt.title("Alpha Mask")

plt.show()
```
```
# Make a copy
faceWithGlassesNaive = faceImage.copy()

# Resize sunglasses
glassResized = cv2.resize(glassBGR, (130, 56))

# Place sunglasses over eyes
x = 142
y = 100

h, w = glassResized.shape[:2]

# Apply sunglasses
faceWithGlassesNaive[y:y+h, x:x+w] = glassResized

# Display
plt.figure(figsize=(5,7))
plt.imshow(cv2.cvtColor(faceWithGlassesNaive, cv2.COLOR_BGR2RGB))
plt.title("Sunglasses Fitted")
plt.axis("on")
plt.show()
```
```

# ==========================================
# STEP 1: LOAD AND RESIZE SUNGLASSES
# ==========================================

# Load original sunglasses image
glassBGR = cv2.imread("Sunglass.png")

if glassBGR is None:
    raise ValueError("Sunglasses image not found!")

# Resize sunglasses
glassBGR = cv2.resize(glassBGR, (145, 63))

# ==========================================
# STEP 2: CREATE MASK
# ==========================================

# Convert sunglasses to grayscale
glassMask1 = cv2.cvtColor(glassBGR, cv2.COLOR_BGR2GRAY)

# Remove white background
_, glassMask1 = cv2.threshold(
    glassMask1, 245, 255, cv2.THRESH_BINARY_INV
)

# Create 3-channel mask
glassMask = cv2.merge((
    glassMask1,
    glassMask1,
    glassMask1
))

# Convert mask values to 0 and 1
glassMask = np.uint8(glassMask / 255)

# ==========================================
# STEP 3: MAKE A COPY OF FACE IMAGE
# ==========================================

faceWithGlasses = faceImage.copy()

# ==========================================
# STEP 4: EYE REGION COORDINATES
# ==========================================

x = 142
y = 100

h, w = glassBGR.shape[:2]

# Extract eye ROI
eyeROI = faceWithGlasses[y:y+h, x:x+w]

# Check dimensions
print("Eye ROI:", eyeROI.shape)
print("Glass Mask:", glassMask.shape)
print("Sunglasses:", glassBGR.shape)

# ==========================================
# STEP 5: MASKING OPERATION
# ==========================================

# Remove region where sunglasses will be placed
maskedEye = cv2.multiply(eyeROI, 1 - glassMask)

# Extract sunglasses using mask
maskedGlass = cv2.multiply(glassBGR, glassMask)

# Combine both images
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Replace eye region with combined image
faceWithGlasses[y:y+h, x:x+w] = eyeRoiFinal

# ==========================================
# STEP 6: DISPLAY INTERMEDIATE RESULTS
# ==========================================

plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(cv2.cvtColor(maskedEye, cv2.COLOR_BGR2RGB))
plt.title("Masked Eye Region")

plt.subplot(132)
plt.imshow(cv2.cvtColor(maskedGlass, cv2.COLOR_BGR2RGB))
plt.title("Masked Sunglass Region")

plt.subplot(133)
plt.imshow(cv2.cvtColor(eyeRoiFinal, cv2.COLOR_BGR2RGB))
plt.title("Combined ROI")

plt.show()

```
```
# Final Result
plt.figure(figsize=(12,8))

plt.subplot(121)
plt.imshow(faceImage[:,:,::-1])
plt.title("Original Image")
plt.axis("on")

plt.subplot(122)
plt.imshow(faceWithGlasses[:,:,::-1])
plt.title("With Sunglasses")
plt.axis("on")
plt.show()
```

### OUTPUT:
### Original Face Image

<img width="438" height="496" alt="image" src="https://github.com/user-attachments/assets/6972402f-561c-450a-be8e-ead92c4b052c" />

### Sunglass Image


<img width="681" height="335" alt="image" src="https://github.com/user-attachments/assets/7657728b-8c4f-44d3-b344-e482cb01ca95" />


### Image Dimension

<img width="397" height="65" alt="image" src="https://github.com/user-attachments/assets/446496ac-5c08-4cb3-8573-36e8397d30ce" />

### Sunglass with colour channel and with alpha channel

<img width="1218" height="251" alt="image" src="https://github.com/user-attachments/assets/cdcf2a4a-d8e6-482d-9faa-45e0e08efcb0" />

### Image with Sunglass

<img width="561" height="655" alt="image" src="https://github.com/user-attachments/assets/b403f3a4-2f0f-4c8d-a780-a3267b70a30b" />

### Augumented Eye and Sunglasses

<img width="1250" height="242" alt="image" src="https://github.com/user-attachments/assets/f16ed113-fb59-42fb-80de-dcf63db07dae" />

### Image with and without sunglasses

<img width="1193" height="686" alt="image" src="https://github.com/user-attachments/assets/fb41746b-b219-4712-b30b-47bce26639ab" />

### RESULT:
Thus, the Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing has been done and verified successfully.
