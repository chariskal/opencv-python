import cv2
import numpy as np
import random
import basic_functions
import imutils


def asRGB (img):
    """
    Function Description: return BGR image as RGB like normal people
    """
    return img[:, :, ::-1]


def adjust_gamma(img, gamma = 1.8):
    """
    Function Description: Gamma correction to image img
    """

    invGamma=1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def draw_delaunay(img, subdiv, delaunay_color):
    """
    Function Description: Draw Delaunay Triangles to image img
    """

    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if basic_functions.is_in_rect(r, pt1) and basic_functions.is_in_rect(r, pt2) and basic_functions.is_in_rect(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

def draw_voronoi(img, subdiv):
    """
    Function Description: Draw Voronoi diagram on image img using subdivisions
    """

    (facets, centers) = subdiv.getVoronoiFacetList([])
    for i in xrange(0,len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.fillConvexPoly(img, ifacet, color, cv2.CV_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.cv.CV_FILLED, cv.CV_AA, 0)


def alpha_blend (aimg, bimg, fimg):
    """
    Function Description: Use alpha blending:
    Args:
            fimg: foreground image
            bimg: background image
            aimg: alpha image
    Returns the blended image
    """
    fimg = fimg.astype(float)               # Convert uint8 to float
    bimg = bimg.astype(float)

    aimg = aimg.astype(float)/255           # Normalize the alpha mask to keep intensity between 0 and 1
    fimg = cv2.multiply(aimg, fimg)         # Multiply the foreground with the alpha matte
    bimg= cv2.multiply(1.0 - aimg, bimg)    # Multiply the background with ( 1 - alpha )

    return cv2.add(fimg, bimg)          # Add the masked foreground and background.



def find_contour_features(img, th = 60, smooth_krn = (5,5), draw =True):
    """
    Function Description: Find and draw contours in an image.  Return centroids
    of contours detected, areas, perimeters, and convex hulls
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                        # img to grayscale
    blurred = cv2.GaussianBlur(gray, smooth_krn, 0)                     # reduce high level noise
    thres = cv2.threshold(blurred, th, 255, cv2.THRESH_BINARY)[1]       # find threshold
    # check to see if we are using OpenCV 2.X or OpenCV 4
    if imutils.is_cv2() or imutils.is_cv4():
        (contours, _) = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    # check to see if we are using OpenCV 3
    elif imutils.is_cv3():
        (_, contours, _) = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    #contours = contours[1:]
    #contours = imutils.grab_contours(contours)
    if draw:
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    centroids = []
    areas = []
    perimeters = []
    convex_hull = []

    for c in contours:
        M = cv2.moments(c)                   # Find image moments for the contour region
        #print(c)
        #print(M)
        cX = int(M["m10"] / basic_functions.zero2one(M["m00"]))
        cY = int(M["m01"] / basic_functions.zero2one(M["m00"]))
        #print(cX,cY)
        #print(int(cX),int(cY))

        centroids = np.append(centroids, (int(cX),int(cY)))
        if draw:
            basic_functions.draw_point(img, (cX,cY))

        a = cv2.contourArea(c)
        areas = np.append(areas, a)

        p = cv2.arcLength(c, True)
        perimeters = np.append(perimeters, p)

        conh = cv2.convexHull(c)
        convex_hull = np.append(convex_hull, conh)

    return img, centroids, areas, perimeters, convex_hull