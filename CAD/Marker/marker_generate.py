import cv2
import numpy as np
import random


# A4 paper dimensions in pixels at 300 DPI (8.27 × 11.69 inches)
A4_WIDTH = 2480
A4_HEIGHT = 3508

scaling_from_print = 1/0.9

MAX_MARK_DIAMETER = 1.3*scaling_from_print # mm
MARKER_SPACING = 3.5*scaling_from_print # mm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def convertMM2Pixel(mm):
    # Convert mm to pixels based on 300 DPI
    MM_TO_PIXELS = 300 / 25.4  # 300 DPI / 25.4 mm per inch
    pixel = mm*MM_TO_PIXELS 

    return int(pixel)

def generateMark(paper, center_x, center_y, is_random=False):
    # Draw a circle on the paper
    circle_radius = convertMM2Pixel(MAX_MARK_DIAMETER/2)

    random_percentage_color = 0.2 
    if is_random and random.random() < random_percentage_color:
        c = random.randint(0, 150)
        color = (c, c, c)
    else:
        color = (0, 0, 0)

    random_percentage = 0.2 
    if is_random and random.random() < random_percentage:
        # Generate random parameters
        radius1 = convertMM2Pixel(random.uniform(0.5*MAX_MARK_DIAMETER, MAX_MARK_DIAMETER)/2)  # Random radius
        radius2 = convertMM2Pixel(random.uniform(0.5*MAX_MARK_DIAMETER, MAX_MARK_DIAMETER)/2)  # Random radius

        start_angle = random.randint(0, 360)  # Random start angle
        end_angle = random.randint(start_angle+180, start_angle + 360)  # Random end angle (to make it incomplete)
        # start_angle = 0
        # end_angle = 360

        # Draw incomplete circle on the black image
        cv2.ellipse(paper, (center_x, center_y), (radius1, radius2), 0, start_angle, end_angle, color, thickness=-1)

    else:
        cv2.circle(paper, (center_x, center_y), circle_radius, color, thickness=-1)

def generateMarks(paper, is_random=False):
    convertMM2Pixel(MARKER_SPACING)
    x_range = np.arange(0, A4_WIDTH, convertMM2Pixel(MARKER_SPACING))
    y_range = np.arange(0, A4_HEIGHT, convertMM2Pixel(MARKER_SPACING))

    centers_xy = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)

    for center_x, center_y in centers_xy:
        generateMark(paper, center_x, center_y, is_random=is_random)

def createPaper():
    # Create a blank A4 sized white image
    paper = np.ones((A4_HEIGHT, A4_WIDTH, 3), dtype=np.uint8) * 255
    generateMarks(paper, is_random=False)

    return paper

if __name__ == "__main__":
    paper = createPaper()

    # Resize the image to display
    # scaled_paper = cv2.resize(paper, (A4_WIDTH // 5, A4_HEIGHT // 5))
    scaled_paper = paper
    
    # Display the scaled image
    cv2.imshow("Circles on A4 Paper (Scaled)", scaled_paper)
    cv2.waitKey(0)
    cv2.imwrite("circles_on_a4_paper_scaled.png", scaled_paper)
    cv2.destroyAllWindows()




# import cv2
# import numpy as np

# # A4 paper dimensions in pixels at 300 DPI (8.27 × 11.69 inches)
# A4_WIDTH = 2480
# A4_HEIGHT = 3508

# # Calculate the radius of the circle to fit within the A4 paper
# circle_radius = min(A4_WIDTH, A4_HEIGHT) // 4

# # Create a blank A4 sized white image
# paper = np.ones((A4_HEIGHT, A4_WIDTH, 3), dtype=np.uint8) * 255

# # Calculate the center of the paper
# center_x = A4_WIDTH // 2
# center_y = A4_HEIGHT // 2

# # Draw a circle on the paper
# cv2.circle(paper, (center_x, center_y), circle_radius, (0, 0, 0), thickness=-1)

# # Resize the image to display
# scaled_paper = cv2.resize(paper, (A4_WIDTH // 6, A4_HEIGHT // 6))

# # Display the scaled image
# cv2.imshow("Circles on A4 Paper (Scaled)", scaled_paper)
# cv2.waitKey(0)
# cv2.imwrite("circles_on_a4_paper_scaled.png", scaled_paper)
# cv2.destroyAllWindows()