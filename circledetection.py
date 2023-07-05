import cv2
import numpy as np
from collections import defaultdict
from math import cos, pi, sin

# set minimum r value
r_min = 20
# set maximum r value
r_max = 30
# read the input file w cv2
input_img = cv2.imread("elevator.png")

# Edge detection on the input image
edge = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
edge = cv2.cvtColor(edge, cv2.COLOR_RGB2GRAY)
# Gaussian Blur on the input image
img_blur = cv2.GaussianBlur(edge, (3, 3), 0)
# I set the min threshold to 100
# I set the max threshold to 200
# You can change the threshold values below if you want
edge = cv2.Canny(img_blur, 100, 200)
# Show the Edge Detection Image
cv2.imshow('Edge Image', edge)
# Save the Edge Detection Image
cv2.imwrite("EdgeImage.jpg", edge)
cv2.waitKey()


def houghCircles(image, edge, min_r, max_r):
    # image size
    img_height = edge.shape[0]
    img_width = edge.shape[1]

    # Radius ranges
    # step one by one
    r_range = np.arange(min_r, max_r, dtype=np.int32)

    circles = []
    for t in range(len(r_range)):
        i = r_range[t]
        for k in range(100):
            circles.append((i, int(i * cos(2 * pi * k / 100)), int(i * sin(2 * pi * k / 100))))

    # I found defaultdict instead of standart dict when i researching
    bus = defaultdict(float)
    # Entering the image as pixel through use x y coordinates (height,width)
    for x in range(img_width):
        for y in range(img_height):
            if edge[y][x] > 0:  # to avoid white pixel
                for a in range(len(circles)):
                    r = circles[a][0]
                    r_cos = circles[a][1]
                    r_sin = circles[a][2]
                    center_x = x - r_cos
                    center_y = y - r_sin
                    bus[(center_x, center_y, r)] += 1

    out_circles = []
    # Items is the property of the defaultdict
    for circles, votes in sorted(bus.items(), key=lambda i: -i[1]):
        x, y, r = circles
        circles_per = votes / 100
        if circles_per >= 0.4:
            out_circles.append((x, y, r, circles_per))

    post_circles = []
    for i in range(len(out_circles)):
        x = out_circles[i][0]
        y = out_circles[i][1]
        r = out_circles[i][2]
        v = out_circles[i][3]
        if all(abs(x - xc) > 10 or abs(y - yc) > 10 or abs(r - rc) > 10 for xc, yc, rc, v in post_circles):
            post_circles.append((x, y, r, v))

    out_circles = post_circles
    output_img = image.copy()
    # Draw circles on the output image
    for i in range(len(out_circles)):
        x = out_circles[i][0]
        y = out_circles[i][1]
        r = out_circles[i][2]
        output_img = cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)

    return output_img


print("Detected Circles Transform Started!")
circle_img = houghCircles(input_img, edge, r_min, r_max)
cv2.imshow('Detected Circles', circle_img)
cv2.waitKey()
cv2.imwrite("DetectedCircles.jpg", circle_img)
print("Detected Circles Transform Complete!")
