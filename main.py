"""https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/"""
"""https://www.tutorialkart.com/opencv/python/opencv-python-get-image-size/"""
#import-urile de care avem nevoie
import cv2
import numpy as np
import argparse
from imutils import paths
from imutils import resize
from imutils import build_montages

#metoda add_argument: construct the argument parse and parse the arguments
#Avem nevoie de o singura linie de comanda ca argument, --images, care este calea catre directorul de imagini
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
args = vars(ap.parse_args())

#functia image_colorfulness: ia imaginea ca singurul argument si o returneaza cu parametrii de analizare a culorii
#pentru pozitionarea parametrilor de culoare s-au folosit formulele:
# rg = R - G
#yb = \frac{1}{2}(R + G) - B
#\sigma_{rgyb} = \sqrt{\sigma_{rg}^2 + \sigma_{yb}^2}
#\mu_{rgyb} = \sqrt{\mu_{rg}^2 + \mu_{yb}^2}
#C = \sigma_{rgyb} + 0.3 * \mu_{rgyb}

def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

#se initializeaza lista pt results si mostColoredImageList
results = []
mostColoredImageList = []

# add custom resolution filter for images
resolutionX = int(input('Resolution for X axis: '))
resolutionY = int(input('Resolution for Y axis: '))

#bucla pentru image path
for imagePath in paths.list_images(args["images"]):
    # load the image, resize it (to speed up computation), and
    # compute the colorfulness metric for the image
    image = cv2.imread(imagePath)
    resizedImage = resize(image, width=250)
    C = image_colorfulness(resizedImage)
    # display the colorfulness score on the images
    cv2.putText(resizedImage, "{:.2f}".format(C), (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.putText(image, "{:.2f}".format(C), (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    # add the image and colorfulness metric to the results list
    results.append((resizedImage, C))
# se iau dimensiunile imaginii; inaltime; latime
    dimensions = image.shape
    height = image.shape[0]
    width = image.shape[1]
#pentru afisarea celei mai colorate imagini
    if len(mostColoredImageList) == 0:
        mostColoredImageList.append((image, C))
    elif height >= resolutionY and width >= resolutionX and C > mostColoredImageList[-1][1]:
        mostColoredImageList.append((image, C))


# construct the montages for the two sets of images
#se sorteaza results reversed
#se stocheaza 25 imagini pt mostColor si 25 pt leastColor
results = sorted(results, key=lambda x: x[1], reverse=True)
mostColor = [r[0] for r in results[:25]]
leastColor = [r[0] for r in results[-25:]][::-1]
mostColoredImage = [r[0] for r in mostColoredImageList[-25:]][::-1]

#se construieste montajul pentru 2 seturi de imagini
#resized 128x128; 5 coloane si 5 randuri
mostColorMontage = build_montages(mostColor, (128, 128), (5, 5))
leastColorMontage = build_montages(leastColor, (128, 128), (5, 5))
#se construieste montajul pentru mostColoredImage
mostColoredImageHeight = mostColoredImage[-1].shape[0]
mostColoredImageWidth = mostColoredImage[-1].shape[1]
mostColoredImageMontage = build_montages(
    mostColoredImage, (mostColoredImageWidth, mostColoredImageHeight), (1, 1))

# se afiseaza imaginile in ferestre separate
#se afiseaza cea mai colorata imagine dupa un anumit prag(introdus de la tastatura)
#waitKey pune pauza executiei programului pana se selecteaza o fereastra activa
cv2.imshow("Most Colorful", mostColorMontage[0])
cv2.imshow("Least Colorful", leastColorMontage[0])
cv2.imshow(
    f"Most Colorful over {resolutionX} x {resolutionY}", mostColoredImageMontage[0])
cv2.waitKey(0)
