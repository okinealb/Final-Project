# === Import Packages =========================================================
import numpy as np              # Importing the NumPy package
from PIL import Image           # Importing Image from the PIL package

# === Functions ===============================================================
# Calculate and return the energy and cost of each pixel
def calculate_energy(row: int, col: int, map: np.ndarray) -> float:
  # If the pixel is on the border, then the corresponding energy is 1000
  if (col == 0 or row == 0 or col == cols-1 or row == rows-1): return 1000
  # Otherwise, calculate the energy of the pixel using the formula given
  else:
    # Calculate the horizontal and vertical costs using the energy function
    h_cost = np.subtract(image_map[row][col-1], image_map[row][col+1], dtype=int)
    v_cost = np.subtract(image_map[row-1][col], image_map[row+1][col], dtype=int)
    # Return the sqrt of the combined, squared horizontal and vertical costs
    return ((np.sum(np.square(h_cost)) + np.sum(np.square(v_cost))) ** (1/2))
  
# Find the minimum cost of the possible pixels above the current pixel
def min_cost(row: int, col: int, cst: np.ndarray) -> int:
  # Determine where the pixel is relative to the border, then find the minimum
  #   cost of the 3 possible pixels immediately above the current pixel
  if   (row == 0): return 0                                  # on top border
  elif (col == 0): return min(cst[row-1][col:col+2])         # on left border
  elif (col == cols-1): return min(cst[row-1][col-1:col+1])  # on right border
  else: return min(cst[row-1][col-1:col+2])                  # inside of border
  
# Remove the pixel (row,col) by first coloring it red, then shifting all pixels
#   after it forward and coloring the last remaining pixels black
def remove_pixel(row: int, col: int, map: np.ndarray, cst: np.ndarray) -> None:
  # Shift the pixels after the initial pixel forward, excluding the pixel given
  map[row][col:cols-1] = map[row][col+1:cols]; map[row][cols-1] = (0,0,0)
  cst[row][col:cols-1] = cst[row][col+1:cols]; cst[row][cols-1] = rows*1000
    
# Find the column index of the next pixel using the previous vertical seam
def find_path(row: int, prev: int, cst: np.ndarray) -> int:
  # Determine where the pixel is relative to the border
  loc = (np.where(cst[row] == min(cst[row][prev : prev+2])) if (prev == 0)
    else np.where(cst[row] == min(cst[row][prev-1:prev+1])) if (prev == cols-1)
    else np.where(cst[row] == min(cst[row][prev-1:prev+2])))
  # Return the column index of the minimum pixel above the previous pixel
  return ([i for i in loc[0] if (abs(i - prev) <= 1)][0])

# === Initialization ==========================================================
# Prompt the user for the input image path and the output image dimensions
file_path = "sample.jpg"
output_width = int(input("Output width: "))
# Use input to import the image, and extract the pixel map as an np array
image = Image.open(file_path)
image_map = np.array(image)

# Initialize cols, rows to the size of the image (width, height, respectively)
rows, cols = image_map.shape[:2]
# Initialize arrays to store energy, table, and seam information into
seam = np.zeros(rows, dtype=int)
energy_tbl = np.zeros((rows, cols))
energy_cst = np.zeros((rows, cols))
# Create output image array to populate later
image_out = np.zeros((rows, output_width, 3), dtype=np.uint8) 

# === Construct Energy Table and Cost =======================================--
# Iterate through each pixel in the image, computing the energy of the pixel
for row in range(rows):
  for col in range(cols):
    # Calculate the energy of the pixel, then update the energy arrays
    energy_tbl[row][col] = calculate_energy(row, col, image_map)
    energy_cst[row][col] = min_cost(row, col, energy_cst) + energy_tbl[row][col]

# === Carve Seams =============================================================
# do better update image instead of moving pixel
for i in range(cols - output_width):
  for row in reversed(range(rows)):
    # Follow the path of the minimum vertical seam, from bottom up
    seam[row] = ((np.where(energy_cst[rows-1] == min(energy_cst[rows-1]))[0][0])
                if row == rows-1 else (find_path(row, seam[row+1], energy_cst)))
    # Remove the pixel (by moving all pixels behind it forward one place)
    remove_pixel(row, seam[row], image_map, energy_cst)
  # Save and output the minimum energy seam to an external csv file
  if (i == 0): np.savetxt("seam1.csv", seam, delimiter=",")

# === Termination =============================================================
# Update the rows in the output image with the rows from the original image
for i in range(rows): image_out[i] = image_map[i][:output_width]
# Save and output image, with each of the seams removed to fit the output width
np.savetxt("energy.csv", energy_tbl[1:rows-1, 1:cols-1], delimiter=",")
(Image.fromarray(image_out)).save("FinalImage.jpg") # Save the new image