"""Utility functions used by digipyro."""
import cv2
import numpy as np
import tkinter
import matplotlib
matplotlib.use("Agg")
import scipy.optimize


# Helper Functions: Section 1 -- User-Interaction Functions
# The majority of functions in this section relate to user-identification of
# the region of interest (ROI) which will be digitally rotated, or the
# intialization of single-particle tracking.
def center_click(event, x, y, flags, param):
    """Allows user to manually identify center of rotation."""
    # declare these variables as global so that they can be used by various
    # functions without being passed explicitly
    global center, frame
    # save the original frame
    clone = frame.copy()
    # if user clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # Set click location as center
        center = (x, y)
        # draw circle at center
        cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        # show updated image
        cv2.imshow('CenterClick', frame)
        # Reset to original image so that if the user reselects the center,
        # the old circle will not appear.
        frame = clone.copy()
    return center, frame


def center_img(img, x_c, y_c):
    """Shift image so that it is centered at (x_c, y_c)"""
    dx = (width / 2) - x_c
    dy = (height / 2) - y_c
    shiftMatrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, shiftMatrix, (width, height))


def unit_conversion(event, x, y, flags, param):
    """User drags mouse and releases to indicate a conversion factor between
    pixels and units of distance.

    """
    global frame, uStart, uEnd, unitCount, unitType, unitConv
    clone = frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        uStart = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        uEnd = (x, y)
        d2 = ((uEnd[0] - uStart[0])**2) + ((uEnd[1] - uStart[1])**2)
        pixelLength = (d2**(0.5))/2
        unitConv = unitCount / pixelLength
        cv2.line(frame, uStart, uEnd, (255, 0, 0), 1)
        cv2.imshow('Distance Calibration', frame)
        frame = clone.copy()


def locate(event, x, y, flags, param):
    """User drags mouse and releases along a diameter of the particle to set an
    approximate size and location of particle for DPR to search for

    """
    # Declare these variables as global so that they can be used by various
    # functions without being passed explicitly
    global frame, particleStart, particleEnd, particleCenter, particleRadius
    clone = frame.copy()							# save the original frame
    if event == cv2.EVENT_LBUTTONDOWN:						# if user clicks
        particleStart = (x,y)							# record location
    elif event == cv2.EVENT_LBUTTONUP:						# if user releases click
        particleEnd = (x,y)							# record location
        particleCenter = ((particleEnd[0] + particleStart[0])//2, (particleEnd[1] + particleStart[1])//2)  # define the center as the midpoint between start and end points
        d2 = ((particleEnd[0] - particleStart[0])**2) + ((particleEnd[1] - particleStart[1])**2)
        particleRadius = (d2**(0.5))/2
        cv2.circle(frame, particleCenter, int(particleRadius+0.5), (255,0,0), 1)	# draw circle that shows the radius and location of cirlces that the Hough circle transform will search for
        cv2.imshow('Locate Ball', frame)					  	# show updated image
        frame = clone.copy() 								# resets to original image


def circumference_points(event, x, y, flags, param):
    """User clicks points along the circumference of a circular ROI. This
    function records the points and calculates the best-fit circle through the
    points.

    """
    global npts, center, frame, xpoints, ypoints, r, poly1, poly2		# declare these variables as global so that they can be used by various functions without being passed explicitly
    if event == cv2.EVENT_LBUTTONDOWN:						# if user clicks
        if (npts == 0):								# if this is the first point, intialize the arrays of x-y coords
            xpoints = np.array([x])
            ypoints = np.array([y])
        else:									# otherwise, append the points to the arrays
            xpoints = np.append(xpoints,x)
            ypoints = np.append(ypoints,y)
        npts+=1
        cv2.circle(frame, (x,y), 3, (0,255,0), -1)
        clone = frame.copy()
        if (len(xpoints) > 2):							# if there are more than 2 points, calculate the best-fit circle through the points
            bestfit = calc_center(xpoints, ypoints)
            center = (bestfit[0], bestfit[1])
            r = bestfit[2]
            poly1 = np.array([[0,0],[frame.shape[1],0],[frame.shape[1],frame.shape[0]], [0,frame.shape[0]]])
            poly2 = np.array([[bestfit[0]+r,bestfit[1]]])
            circpts = 100
            for i in range(1,circpts):						# approximate the circle as a 100-gon (which makes it easier to draw the mask, as we define the mask region as the area between two polygons)
                theta =  2*np.pi*(float(i)/circpts)
                nextpt = np.array([[int(bestfit[0]+(r*np.cos(theta))),int(bestfit[1]+(r*np.sin(theta)))]])
                poly2 = np.append(poly2,nextpt,axis=0)
            cv2.circle(frame, center, 4, (255,0,0), -1)
            cv2.circle(frame, center, r, (0,255,0), 1)
        cv2.imshow('CenterClick', frame)
        frame = clone.copy()


def n_gon(event, x, y, flags, param):
    """The same as "circumference_points", except this calculates a polygon
    ROI. The center is calculated as the "center of mass" of the polygon

    """
    global npts, center, frame, xpoints, ypoints, r, poly1, poly2 		# declare these variables as global so that they can be used by various functions without being passed explicitly
    if event == cv2.EVENT_LBUTTONDOWN:						# if user clicks
        if (npts == 0):
            xpoints = np.array([x])
            ypoints = np.array([y])
        else:
            xpoints = np.append(xpoints,x)
            ypoints = np.append(ypoints,y)
        npts+=1
        cv2.circle(frame, (x,y), 3, (0,255,0), -1)
        clone = frame.copy()
        if (len(xpoints) > 2):
            center = (int(np.sum(xpoints)/npts), int(np.sum(ypoints)/npts))
            poly1 = np.array([[0,0],[frame.shape[1],0],[frame.shape[1],frame.shape[0]], [0,frame.shape[0]]])
            poly2 = np.array([[xpoints[0],ypoints[0]]])
            for i in range(len(xpoints)-1):
                nextpt = np.array([[xpoints[i+1], ypoints[i+1]]])
                poly2 = np.append(poly2,nextpt,axis=0)
                cv2.line(frame, (xpoints[i], ypoints[i]), (xpoints[i+1], ypoints[i+1]), (0, 255, 0), 1)
            cv2.line(frame, (xpoints[len(xpoints)-1], ypoints[len(xpoints)-1]), (xpoints[0],ypoints[0]), (0, 255, 0), 1)
            cv2.circle(frame, center, 4, (255,0,0), -1)
        cv2.imshow('CenterClick', frame)
        frame = clone.copy()


def remove_point(orig):
    """Removes the most recently clicked point in the array of circle/polygon
    circumference points.

    """
    global npts, center, frame, xpoints, ypoints, r, poly1, poly2, custMask
    if npts == 0:
        return

    else:
        npts -= 1
        if npts == 0:
            xpoints = np.empty(0)
            ypoints = np.empty(0)
        elif npts == 1:
            xpoints = np.array([xpoints[0]])
            ypoints = np.array([ypoints[0]])
        else:
            xpoints = xpoints[0:npts]
            ypoints = ypoints[0:npts]

    frame = orig.copy()
    for i in range(len(xpoints)):
        cv2.circle(frame, (xpoints[i], ypoints[i]), 3, (0,255,0), -1)
    if (len(xpoints) > 2):							# if there are more than 2 points after removing the most recent point, recalculate the center of rotation and the mask region
        if custMask:
            poly1 = np.array([[0,0],[frame.shape[1],0],[frame.shape[1],frame.shape[0]], [0,frame.shape[0]]])
            poly2 = np.array([[xpoints[0],ypoints[0]]])
            for i in range(len(xpoints)-1):
                nextpt = np.array([[xpoints[i+1], ypoints[i+1]]])
                poly2 = np.append(poly2,nextpt,axis=0)
                cv2.line(frame, (xpoints[i], ypoints[i]), (xpoints[i+1], ypoints[i+1]), (0, 255, 0), 1)
            cv2.line(frame, (xpoints[len(xpoints)-1], ypoints[len(xpoints)-1]), (xpoints[0],ypoints[0]), (0, 255, 0), 1)
            cv2.circle(frame, center, 4, (255,0,0), -1)
        else:
            bestfit = calc_center(xpoints, ypoints)
            center = (bestfit[0], bestfit[1])
            r = bestfit[2]
            poly1 = np.array([[0,0],[frame.shape[1],0],[frame.shape[1],frame.shape[0]], [0,frame.shape[0]]])
            poly2 = np.array([[bestfit[0]+r,bestfit[1]]])
            circpts = 100
            for i in range(1,circpts):
                theta =  2*np.pi*(float(i)/circpts)
                nextpt = np.array([[int(bestfit[0]+(r*np.cos(theta))),int(bestfit[1]+(r*np.sin(theta)))]])
                poly2 = np.append(poly2,nextpt,axis=0)
            cv2.circle(frame, center, 4, (255,0,0), -1)
            cv2.circle(frame, center, r, (0,255,0), 1)
        cv2.imshow('CenterClick', frame)


def calc_center(xp, yp):
    """Calculates the center and radius of the best-fit circle through an array
    of points (by least-squares method).

    """
    n = len(xp)
    circleMatrix = np.matrix([[np.sum(xp**2), np.sum(xp*yp), np.sum(xp)], [np.sum(xp*yp), np.sum(yp**2), np.sum(yp)], [np.sum(xp), np.sum(yp), n]])
    circleVec = np.transpose(np.array([np.sum(xp*((xp**2)+(yp**2))), np.sum(yp*((xp**2)+(yp**2))), np.sum((xp**2)+(yp**2))]))
    ABC = np.transpose(np.dot(np.linalg.inv(circleMatrix), circleVec))
    xc = ABC.item(0)/2
    yc = ABC.item(1)/2
    a = ABC.item(0)
    b = ABC.item(1)
    c = ABC.item(2)
    d = (4*c)+(a**2)+(b**2)
    diam = d**(0.5)
    return np.array([int(xc), int(yc), int(diam/2)])


def annotate_img(img, i):
    """Adds diagnostic information, including time and physical/digital
    rotations to each frame of the movie.

    """
    font = cv2.FONT_HERSHEY_TRIPLEX

    dpro = 'DigiPyRo'
    dproLoc = (25, 50)
    cv2.putText(img, dpro, dproLoc, font, 1, (255, 255, 255), 1)

    img[25:25+spinlab.shape[0], (width-25)-spinlab.shape[1]:width-25] = spinlab

    timestamp = 'Time: ' + str(round((i/fps),1)) + ' s'
    tLoc = (width - 225, height-25)
    cv2.putText(img, timestamp, tLoc, font, 1, (255, 255, 255), 1)

    crpm = 'Original Rotation of Camera: '
    crpm += str(cam_rpm) + 'RPM'

    drpm = 'Additional Digital Rotation: '
    if (digi_rpm > 0):
        drpm += '+'
    drpm += str(digi_rpm) + 'RPM'

    cLoc = (25, height - 25)
    dLoc = (25, height - 65)
    cv2.putText(img, drpm, dLoc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, crpm, cLoc, font, 1, (255, 255, 255), 1)


def annotate_sbs(img):
    font = cv2.FONT_HERSHEY_TRIPLEX

    orig = 'Raw Movie'
    origLoc = (int(0.01*r), 50)

    dpred = 'DigiPyRo: '
    if (digi_rpm > 0):
        dpred += '+'
    dpred += str(digi_rpm) + 'RPM'
    dpredLoc = (int(2.01*r)+30, 50)

    cv2.putText(img, orig, origLoc, font, 2, (255, 255, 255), 1)
    cv2.putText(img, dpred, dpredLoc, font, 2, (255, 255, 255), 1)

    img[img.shape[0] - (spinlab.shape[0] + 25): img.shape[0] - 25, img.shape[1] - (spinlab.shape[1] + 25): img.shape[1] - 25] = spinlab


def instructs_center(img):
    """Display instructions on the screen for identifying the circle/polygon
    of interest.
    """
    font = cv2.FONT_HERSHEY_PLAIN
    line1 = 'Click on 3 or more points along the border of the circle or polygon'
    line1Loc = (25, 50)
    line2 = 'around which the movie will be rotated.'
    line2Loc = (25, 75)
    line3 = 'Press the BACKSPACE or DELETE button to undo a point.'
    line3Loc = (25, 100)
    line4 = 'Press ENTER when done.'
    line4Loc = (25, 125)

    cv2.putText(img, line1, line1Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line2, line2Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line3, line3Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line4, line4Loc, font, 1, (255, 255, 255), 1)


def instructs_ball(img, font=None):
    """Display instructions for drawing a circle around the ball."""
    if font is None:
        font = cv2.FONT_HERSHEY_PLAIN
    line1 = 'Click and drag to create a circle around the ball.'
    line1Loc = (25, 50)
    line2 = 'The more accurately the initial location and size of the ball'
    line2Loc = (25, 75)
    line3 = 'are matched, the better the tracking results will be.'
    line3Loc = (25, 100)
    line4 = 'Press ENTER when done.'
    line4Loc = (25, 125)

    cv2.putText(img, line1, line1Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line2, line2Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line3, line3Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line4, line4Loc, font, 1, (255, 255, 255), 1)


def instructs_unit(img, font=None):
    """Display instructions for drawing a line for unit conversion."""
    if font is None:
        font = cv2.FONT_HERSHEY_PLAIN
    line1 = 'Click and release to draw a line of'
    line1Loc = (25, 50)
    line2 = str(unitCount) + ' ' + unitType
    line2Loc = (25, 75)
    line3 = 'Press ENTER when done.'
    line3Loc = (25, 100)

    cv2.putText(img, line1, line1Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line2, line2Loc, font, 1, (255, 255, 255), 1)
    cv2.putText(img, line3, line3Loc, font, 1, (255, 255, 255), 1)


# Mathematical Utility Functions
def calc_deriv(f, t):
    """2nd-order central difference method for calculating the derivative
    of unevenly spaced data.

    """
    df = np.empty(len(f))
    df[0] = (f[1] - f[0]) / (t[1] - t[0])
    df[len(f)-1] = (f[len(f)-1] - f[len(f)-2]) / (t[len(f)-1] - t[len(f)-2])
    df[1:len(f)-1] = (
        f[0:len(f)-2]*((t[1:len(f)-1] - t[2:len(f)]) /
                       ((t[0:len(f)-2] - t[1:len(f)-1])*(t[0:len(f)-2] -
                                                         t[2:len(f)]))) +
        f[1:len(f)-1]*(((2*t[1:len(f)-1]) - t[0:len(f)-2] - t[2:len(f)]) /
                       ((t[1:len(f)-1] - t[0:len(f)-2])*(t[1:len(f)-1] -
                                                         t[2:len(f)]))) +
        f[2:len(f)]*((t[1:len(f)-1] - t[0:len(f)-2]) /
                     ((t[2:len(f)] - t[0:len(f)-2])*(t[2:len(f)] -
                                                     t[1:len(f)-1])))
    )
    return df


def spline_fit(x, y, deg):
    """Calculate a polynomial fit of degree "deg" though an array of data "y"
    with corresponding x values "x".
    """
    fit = np.polyfit(x, y, deg)
    yfit = np.zeros(len(y))
    for i in range(deg+1):
        yfit += fit[i]*(x**(deg-i))
    return yfit


# The following functions assist in estimating the coefficient of friction of
# the user's table by fitting their data to a damped harmonic oscillator. This
# functionality is not implemented in the current release of DigiPyRo
def err_func_polar(params, data):
    modelR = np.abs(params[0]*np.exp(-data[0]*params[3]*params[1]) *
                    np.cos((params[3]*data[0] *
                            ((1-(params[1]**2))**(0.5))) - params[2]))
    modelTheta = create_model_theta(data[0], params, data[2][0])
    model = np.append(modelR, modelR*modelTheta)
    datas = np.append(data[1], data[1]*data[2])
    return model - datas


def fit_data_polar(data, guess):
    return scipy.optimize.leastsq(err_func_polar, guess, args=(data),
                                  full_output=1)[0]


def create_model_r(bestfit, t):
    return np.abs(bestfit[0] * np.exp(-t*bestfit[3] * bestfit[1]) *
                  np.cos((bestfit[3] * t * ((1 - (bestfit[1]**2))**(0.5)) -
                          bestfit[2])))


def create_model_theta(t, bestfit, thetai):
    wd = bestfit[3] * ((1 - (bestfit[1])**2)**(0.5))
    phi = bestfit[2]
    theta = np.ones(len(t))*thetai
    for i in range(len(t)):
        phase = (wd*t[i])-phi
        while phase > 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi

        if phase < (np.pi/2) or phase > ((3*np.pi)/2):
            theta[i] = thetai
        elif phase > (np.pi/2) and phase < ((3*np.pi)/2):
            theta[i] = thetai + np.pi
        theta[i] += t[i]*(-bestfit[4])

        while theta[i] > 2*np.pi:
            theta[i] -= 2*np.pi
        while theta[i] < 0:
            theta[i] += 2*np.pi
    return theta


if __name__ == "__main__":
    pass
