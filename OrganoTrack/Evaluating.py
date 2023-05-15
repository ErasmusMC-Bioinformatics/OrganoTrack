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



def EvaluateSegmentationAccuracy(predictionImage, groundTruthImage):

    predictionBinary_1, groundTruthBinary_1 = BinariseTo1(predictionImage, groundTruthImage)

    # Count true positives, false positives and false negatives
    sumImage = cv.add(predictionBinary_1, groundTruthBinary_1)  # 0 or 1 or 2
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
    # Convert ground truth image to RGB green
    groundTruthRGB = cv.cvtColor(groundTruthImage, cv.COLOR_GRAY2RGB)
    _, groundTruthRGB = cv.threshold(groundTruthRGB, 50, 255, cv.THRESH_BINARY)
    groundTruthRGB[np.all(groundTruthRGB == (255, 255, 255), axis=-1)] = (0, 255, 0)

    # Convert prediction image to RGB
    predictionRGB = cv.cvtColor(predictionImage, cv.COLOR_GRAY2RGB)

    # Generating overlay
    alpha = 0.5
    beta = 1 - alpha
    overlay = cv.addWeighted(predictionRGB, alpha, groundTruthRGB, beta, 0.0)


    return scores, overlay

def Test_Evaluate():
    gtImageDir = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/annotated/annotations/images/d0r1t0_GT.png'
    groundTruthImage = cv.imread(gtImageDir, cv.IMREAD_GRAYSCALE)

    predImageDir = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictions/segmented-10.05.2023-15_03_26/d0r1t0.tiff'
    predImage = cv.imread(predImageDir, cv.IMREAD_GRAYSCALE)

    exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictions')
    predImagePath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictions/segmented-10.05.2023-15_03_26/d0r1t0.tiff')
    saveImgOverlay = [True, exportPath, predImagePath]

    scores, overlay = EvaluateSegmentationAccuracy(predImage, groundTruthImage)

    print(scores)
    Display('overlay', overlay, 0.5)
    cv.waitKey(0)

if __name__ == '__main__':
    Test_Evaluate()