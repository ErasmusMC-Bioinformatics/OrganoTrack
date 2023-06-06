import numpy as np
import cv2 as cv
from pathlib import Path
from OrganoTrack.Displaying import Display

def BinariseTo1(predictionImage, groundTruthImage):

    predictionGrayscale = cv.cvtColor(predictionImage, cv.COLOR_GRAY2BGR)
    _, predictionBinary_255 = cv.threshold(predictionGrayscale, 10, 255, cv.THRESH_BINARY)
    predictionBinary_1 = predictionBinary_255 / 255

    groundTruthGrayscale = cv.cvtColor(groundTruthImage, cv.COLOR_GRAY2BGR)
    _, groundTruthBinary_255 = cv.threshold(groundTruthGrayscale, 10, 255, cv.THRESH_BINARY)
    groundTruthBinary_1 = groundTruthBinary_255 / 255

    return predictionBinary_1, groundTruthBinary_1

def ColouriseImage(image, colour):
    image = image.astype(np.uint8)
    colourCodes = {'blue': (219, 152, 52),
                   'orange': (34, 126, 230),
                   'green': (96, 174, 39),
                   'gray': (199, 195, 189),
                   'red': (60, 76, 231)}

    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY)

    if len(image.shape) != 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    else:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image[np.all(image == (255, 255, 255), axis=-1)] = colourCodes[colour]

    return image

def EvaluateSegmentationAccuracy(predictionImage, groundTruthImage):
    displayScale = 0.5
    _, groundTruthImage = cv.threshold(groundTruthImage, 10, 255, cv.THRESH_BINARY)

    predictionBinary_1, groundTruthBinary_1 = BinariseTo1(predictionImage, groundTruthImage)

    # Count true positives, false positives and false negatives
    sumImage = cv.add(predictionBinary_1, groundTruthBinary_1)  # 0 or 1 or 2
    truePositiveImage = cv.bitwise_and(predictionImage, groundTruthImage)

    truePositiveCount = np.count_nonzero(sumImage == 2)

    orImage = cv.bitwise_or(predictionBinary_1, groundTruthBinary_1)  # 0 or 1, not 2

    falsePositiveImage = cv.subtract(orImage, groundTruthBinary_1)  # 0 or 1, not 2
    falsePositiveCount = np.count_nonzero(falsePositiveImage == 1)

    falseNegativeImage = cv.subtract(orImage, predictionBinary_1)  # 0 or 1, not 2
    falseNegativeCount = np.count_nonzero(falseNegativeImage == 1)

    # Calculate scores
    f1Score = 100*2 * truePositiveCount / (2 * truePositiveCount + falsePositiveCount + falseNegativeCount)
    iouScore = 100*truePositiveCount/np.count_nonzero(orImage == 1)
    diceScore = 100*2*truePositiveCount/(np.count_nonzero(predictionBinary_1 == 1) + np.count_nonzero(groundTruthBinary_1 == 1))
    scores = np.array([f1Score, iouScore, diceScore])

    truePositiveColour = ColouriseImage(truePositiveImage, 'blue')
    falsePositiveColour = ColouriseImage(255*falsePositiveImage, 'orange')
    falseNegativeColour = ColouriseImage(255*falseNegativeImage, 'red')

    overlay = cv.addWeighted(truePositiveColour, 1, falsePositiveColour, 1, 0)
    overlay = cv.addWeighted(overlay, 1, falseNegativeColour, 1, 0)

    return scores, overlay

def Test_Evaluate():
    gtImageDir = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/groundTruth/images/d0r1t3_GT.png'
    groundTruthImage = cv.imread(gtImageDir, cv.IMREAD_GRAYSCALE)

    predImageDir = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictions/OrganoTrack-segmented/images/d0r1t3.tiff'
    predImage = cv.imread(predImageDir, cv.IMREAD_GRAYSCALE)

    scores, overlay = EvaluateSegmentationAccuracy(predImage, groundTruthImage)

    print(scores)
    Display('overlay', overlay, 0.5)
    cv.waitKey(0)

if __name__ == '__main__':
    Test_Evaluate()