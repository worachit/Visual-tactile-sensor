import numpy as np
import cv2

file_path = "./Serial Debug 2024-3-20 1374.txt"


def convert_rgb565_to_rgb(rgb565_array):
    red = 255.0*((rgb565_array >> 11) & 0x1F)/float(0x1F)
    green = 255.0*((rgb565_array >> 5) & 0x3F)/float(0x3F)
    blue = 255.0*(rgb565_array & 0x1F)/float(0x1F)   

    # Combine R, G, B components into RGB image
    rgb_image = np.dstack((red, green, blue))

    return rgb_image.astype(np.uint8)

width = 320
height = 240
DATA_SIZE = width*height*2
with open(file_path, "r") as f:
    content = f.read()
    separated_content = content.split()

    data = []
    if len(separated_content) == DATA_SIZE:
        print("correct data")
        for i in range(0,DATA_SIZE,2):
            number = int(separated_content[i] + separated_content[i+1], 16)
            data.append(number)

        data = np.array(data, dtype=np.uint16)
        rgb_image = convert_rgb565_to_rgb(data)

        # Reshape the image to its original dimensions
        rgb_image = np.reshape(rgb_image, (height, width, 3))
        
        rgb_image = cv2.resize(rgb_image, (height*2, width*2))
        # Display the image using OpenCV
        cv2.imshow('RGB565 Image', rgb_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()