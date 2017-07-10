import cv2;

def showImage(imgfilename):
    print "zl's open image function"
    print "loading image %s..." % imgfilename
    image = cv2.imread(imgfilename)

    shap = image.shape
    print shap

    cv2.imshow('image show', image)
    cv2.waitKey(5 * 1000)
    cv2.destroyAllWindows()

showImage("bird.jpg")
