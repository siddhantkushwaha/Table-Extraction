from PIL import Image
import cv2 as cv

from extract import extract

if __name__ == '__main__':
    ext_img = Image.open('data/example.jpg')
    ext_img.save("out/target.jpg", "JPEG")
    target_img = cv.imread("out/target.jpg")

    tables = extract(target_img)
