import serial
import numpy as np
import cv2

width = 320
height = 240
DATA_SIZE = width*height*2

def convert_rgb565_to_rgb(rgb565_array):
    red = 255.0*((rgb565_array >> 11) & 0x1F)/float(0x1F)
    green = 255.0*((rgb565_array >> 5) & 0x3F)/float(0x3F)
    blue = 255.0*(rgb565_array & 0x1F)/float(0x1F)  

    # Combine R, G, B components into RGB image
    rgb_image = np.dstack((red, green, blue))

    return rgb_image.astype(np.uint8)

if __name__ == "__main__":
    # configure the serial connections
    ser = serial.Serial(
        port='COM8',
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS, 
        timeout=30000
    )
    ser.set_buffer_size(rx_size = int(5*DATA_SIZE/4), tx_size = int(5*DATA_SIZE/4))

    ser.isOpen()
    data = ser.read(int(DATA_SIZE)) 
    ser.close()

    data_list = list(data)
    new_data = []
    if len(data_list) == DATA_SIZE:
        print("correct data")
        for i in range(0,DATA_SIZE,2):
            number = ((data[i] << 8) & 0xFF00) | (data[i+1] & 0x00FF) 
            new_data.append(number)
        
        new_data = np.array(new_data, dtype=np.uint16)
        rgb_image = convert_rgb565_to_rgb(new_data)

        # Reshape the image to its original dimensions
        rgb_image = np.reshape(rgb_image, (height, width, 3))
        
        rgb_image = cv2.resize(rgb_image, (height*2, width*2))
        # Display the image using OpenCV
        cv2.imshow('RGB565 Image', rgb_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()