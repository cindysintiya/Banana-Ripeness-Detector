import rembg
import numpy as np
from PIL import Image, ImageOps
import cv2

def remove_background(img: Image.Image) -> Image.Image:
  """
    Accept an Image object, remove the background, and returns Image with removed background
  """
  input_array = np.array(img)

  # Apply background removal using rembg
  output_array = rembg.remove(input_array)

  # Create a PIL Image from the output array
  output_image = Image.fromarray(output_array)

  return output_image

def resize_with_padding(img: Image.Image, expected_size: tuple[int,int]) -> Image.Image:
  """
    Resize the Image object to the expected size (width, height); adding zero padding to fill empty pixels
  """
  img.thumbnail((expected_size[0], expected_size[1]))
  delta_width = expected_size[0] - img.size[0]
  delta_height = expected_size[1] - img.size[1]
  pad_width = delta_width // 2
  pad_height = delta_height // 2
  padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
  return ImageOps.expand(img, padding)

def create_binary_mask(img: Image) -> Image:
  """
    Accept an Image object, return the binary Image
  """
  img = np.array(img)
  gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  blur = cv2.GaussianBlur(gray_image, (11,11), 0)
  et,th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY)
  # th1 = cv2.bitwise_not(th1)

  return Image.fromarray(th1)

# Edge detection with Canny
def edge_detection(image: Image.Image) -> list[Image.Image, float]:
  '''
    Accept Image object in grayscale mode, return a list containing\n
    the edge detection Image result using Canny filter and edge strength
  '''
  gray_img = np.array(image)

  edges = cv2.Canny(gray_img, 150, 200)  # 100, 200
  edge_strength = np.sum(edges) / 255.0

  # Stack the edges back into 3 channels to maintain shape consistency
  edges_stacked = np.stack((edges,)*3, axis=-1)

  edges_img = Image.fromarray(edges_stacked)

  return edges_img, edge_strength

def calculate_edge_density(edge_map: cv2.typing.MatLike, roi_mask: cv2.typing.MatLike) -> float:
  '''
    Calculate edge density within the region of interest (ROI)
  '''

  roi_edges = cv2.bitwise_and(edge_map, edge_map, mask=roi_mask)
  # plt.imshow(roi_edges)
  edge_pixels = np.sum(roi_edges) / 255  # Count edge pixels (each pixel is 255 in binary)
  total_pixels = np.sum(roi_mask) / 255  # Count pixels in the ROI

  if total_pixels == 0:  # Avoid division by zero
    return 0
  edge_density = edge_pixels / total_pixels
  return edge_density

def getLargestContour(img_array: cv2.typing.MatLike) -> cv2.typing.MatLike:
  '''
    Calculate the largest contour (bounding box)
  '''
  des = img_array
  contour, hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

  for cnt in contour:
      cv2.drawContours(des,[cnt],0,255,-1)

  largest_contour = max(contour, key=cv2.contourArea)
  return largest_contour

def resize_to_bounding_box(image: Image.Image, largest_contour: cv2.typing.MatLike) -> Image.Image:
  '''
    Crop according to the largest contour (bounding box),\n
    then resize it to 128x128
  '''
  img_array = np.array(image)
  x, y, w, h = cv2.boundingRect(largest_contour)
  banana_roi = img_array[y:y+h, x:x+w]
  banana_roi = Image.fromarray(banana_roi)
  banana_roi_resized = resize_with_padding(banana_roi, (128,128))
  return banana_roi_resized

def create_mask(image_rgb: cv2.typing.MatLike, lower: list[int,int,int], upper: list[int,int,int]) -> cv2.typing.MatLike:
  '''
    Accept np.array-ed RGB Image, HSV color low range, HSV color upper range\n
    Return array of masked Image
  '''
  # Ensure image_rgb is 3-channel RGB (png has 4 channel, RGBA)
  if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
    # Convert to 3-channel RGB if necessary
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

  # To handle opencv manipulations, need to store in bgr bcs opencv processed images in bgr mode
  image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
  # Convert the image to HSV color
  image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

  # Create a mask where the object is 1 and background is 0
  mask = cv2.inRange(image_hsv, lower, upper)
  # ax[0].imshow(mask)
  # ax[0].set_title('Mask')

  final = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
  masked_img = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
  # ax[1].imshow(masked_img)
  # ax[1].set_title('Masked Image')

  # ax[2].imshow(image_rgb)
  # ax[2].set_title('Original Image')
  # plt.show()
  return masked_img

def create_yellow_mask(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
  '''
    Accepts np.array-ed RGB Image, return array of yellow-masked Image
  '''
  # Define range of yellow banana colors in HSV (this will need fine-tuning)
  lower_yellow = np.array([15, 100, 100])
  upper_yellow = np.array([30, 255, 255])
  return create_mask(image, lower_yellow, upper_yellow)

def create_green_mask(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
  '''
    Accepts np.array-ed RGB Image, return array of green-masked Image
  '''
  # Define range of green banana colors in HSV (this will need fine-tuning)
  lower_green = np.array([35, 70, 70])
  upper_green = np.array([75, 255, 255])
  return create_mask(image, lower_green, upper_green)

def preprocess_input(img: Image.Image) -> list[cv2.typing.MatLike]:
  '''
    Preprocess image from user.\n
    Accepts PIL Image, returns (128,128,9) image (yellow mask, green mask, edge)
  '''
  img = resize_with_padding(img, (500,500))
  img_clean = remove_background(img)

  img_binary = create_binary_mask(img_clean)
  contour = getLargestContour(np.array(img_binary))

  # Create Green and Yellow Mask
  img_bounding_box = resize_to_bounding_box(img, contour)
  arr = np.array(img_bounding_box)
  yellow_mask = create_yellow_mask(arr)
  green_mask = create_green_mask(arr)

  # Edge Detection
  img_binary_roi = resize_to_bounding_box(img_binary, contour)
  img_edge, edge_strength = edge_detection(img_binary_roi)

  # Compute edge strength and density (boleh otak-atik manatau ada yg gak kepake)
  roi_mask = np.array(img_binary_roi)
  roi_mask = cv2.bitwise_not(roi_mask)
  edge_density = calculate_edge_density(np.array(img_edge), roi_mask) * 100

  # fine tuning needed
  # edge_density_threshold_low = 15
  # edge_density_threshold_high = 35
  edge_density_threshold_low = 5.59 - 0.25
  edge_density_threshold_high = 12.58 + 0.25
  if edge_density < edge_density_threshold_low or edge_density > edge_density_threshold_high:
    print("Classified as Not a Banana based on edge density.")
    return [[yellow_mask, green_mask, img_edge]]  # Skip further processing for not-a-banana

  combined_image = np.dstack((yellow_mask, green_mask, img_edge))
  print('Density', edge_density)
  # print(np.array(yellow_mask).max())
  # print(np.array(green_mask).max())
  # print(np.array(img_edge).max())
  return [[yellow_mask, green_mask, img_edge], combined_image]