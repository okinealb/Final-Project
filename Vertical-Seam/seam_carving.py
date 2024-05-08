# === Import Packages =========================================================
import numpy as np              # Importing the NumPy package
from PIL import Image           # Importing Image from the PIL package

# === Functions ===============================================================
# Calculate the energy of the pixel (x,y) by considering the RGB values of the
#   pixels immediately above/below and to the left/right.
def calculate_energy(x, y):
  # If the pixel is on the border, then the corresponding energy is 1000
  if (x == 0 or y == 0 or x == cols-1 or y == rows-1): return 1000
  # Otherwise, calculate the energy of the pixel (x,y) using the formula
  else:
    # Calculate the horizontal and vertical costs using the energy function
    h_cost = np.subtract(image.getpixel((x-1, y)), image.getpixel((x+1, y)))
    v_cost = np.subtract(image.getpixel((x, y-1)), image.getpixel((x, y+1)))
    # Return the sqrt of the combined, squared horizontal and vertical costs 
    return (np.sum(np.square(h_cost)) + np.sum(np.square(v_cost))) ** 1/2
  
# Find the minimum cost of the possible pixels above the current pixel at (x,y)
def min_cost(tbl, x, y):
  # Determine where the pixel is relative to the border, changing according to
  if   (y == 0): return 0                             # on top border
  elif (x == 0): return min(tbl[y-1][x:x+2])          # on left border
  elif (x == cols-1): return min(tbl[y-1][x-1:x+1])   # on right border
  else: return min(tbl[y-1][x-1:x+2])                 # inside of border
  
# Remove the pixel (row,col) by first coloring it red, then shifting all pixels
#   after it forward and coloring the last remaining pixels black
def remove_pixel(pixels, tbl, x, y):
  color_red(pixels, x, y) 
  # Shift the pixels after the initial pixel (col,row) forward
  for col in range(x, cols): 
    # Shift all pixels behind forward, then olor all remaining pixels black
    pixels[col, y] =  pixels[col + 1, y] if (col < cols - 1) else (0, 0, 0)
    tbl[y][col] = tbl[y][col + 1] if (col < cols - 1) else rows * 1000
    
# Indicate which pixels are on the seam by coloring the pixel (col,row) red
def color_red(pixels, x, y): 
  # Return the new pixel_map, with the pixel (col,row) colored
  pixels[x, y] = (255, 0, 0)

# Find the index of the minimum energy in the row
def min_index(row):
  # First find the minimum energy seam in the row
  min_energy = min(energy_cst[row])
  # Iterate through the energy table to find the index of the minimum
  for col in range(cols):
    if (energy_cst[row][col] == min_energy): return col
    
# Find the column index of the next pixel using the previous vertical seam
def find_path(tbl, row, prev):
  # Determine whether the previous seam is on the border or not
  if (prev == 0):  
    loc = np.where(tbl[row] == min(tbl[row][prev : prev+2]))
  elif (prev == cols-1):
    loc = np.where(tbl[row] == min(tbl[row][prev-1:prev+1]))
  else: 
    loc = np.where(tbl[row] == min(tbl[row][prev-1:prev+2]))
  # Return the column index of the minimum pixel above the previous pixel
  return find_nearest(loc[0], prev)
    

def find_nearest(arr, index): 
  for i in range(len(arr)): 
    if (abs(arr[i] - index) <= 1): return arr[i]
  

# === Initialization ==========================================================
# Read in the input image of the ocean
file_path = "sample.jpg"

# Prompt the user for the input image path and the dimensions of the output
#   image, then store them
output_width = 408
#output_w = int(input("Output width: "))    # Translate the output width into an int

image = Image.open(file_path) # Import the image from the given filepath
pixel_map = image.load()      # Extract the pixel map from input

# Initialize cols, rows to the size of the image (width, height, respectively)
cols, rows = image.size
# Initialize arrays to store energy, table, and seam information into
seam = np.zeros(rows, dtype=int)
energy_cst = np.zeros((rows, cols), dtype=int)
energy_tbl = np.zeros((rows, cols), dtype=int)

# === Construct Energy Costs and Tables =======================================
# Iterate through each pixel in the image, computing the energy of the pixel
for row in range(rows):
  for col in range(cols):
    # Calculate the energy of the pixel (col,row), then update the energy arrays
    energy_tbl[row][col] = calculate_energy(col, row)
    energy_cst[row][col] = min_cost(energy_cst, col, row) + energy_tbl[row][col]
    
# === Carve Seams =============================================================
for i in range(200):
  for row in reversed(range(rows)):
    if (row == rows-1):
      # Find the current minimum vertical seam in the picture
      seam[row] = np.where(energy_cst[rows-1] == min(energy_cst[rows-1]))[0][0]
    else:
      # Follow the path of the minimum vertical seam, from bottom up
      seam[row] = find_path(energy_cst, row, seam[row+1])
    # Remove the pixel (by moving all pixels behind it forward one place)
    color_red(pixel_map, seam[row], row)
  image.save(f"Image{i}.jpg")
  for row in reversed(range(rows)):
    remove_pixel(pixel_map, energy_cst, seam[row], row)
  #image.save(f"Image{i}.jpg")


# === Termination =============================================================
# Save and output the energies, smallest weight seam, and final image
np.transpose(energy_tbl).tofile("energy.csv", sep = ',') # Write pixel energies
np.transpose(seam).tofile("seam1.csv", sep = ',') # Write the smallest seam
image.save("FinalImage.jpg") # Save the output image, with the seams removed

print(seam)