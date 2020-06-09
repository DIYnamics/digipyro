#! /usr/bin/env python
"""
DigiPyRo is a program with two main functions:
1. Digital rotation of movies.
2. Single-particle tracking.
All of its functionalities can be accessed through the GUI window which appears
when DigiPyRo is run.  See the README and instructables for further
documentation, installation instructions, and examples.

"""
import time.strftime

import cv2
import numpy as np
from tkinter import (
    BooleanVar,
    Button,
    Checkbutton,
    DoubleVar,
    Entry,
    Label,
    StringVar,
    Tk,
)
import matplotlib
from matplotlib import pyplot as plt


matplotlib.use("Agg")


class Digipyro(object):
    def __init__(self):
        self.filename = None
        self.fps = None

    def start(self):
        """Executed when the user presses the "Start!" button on the GUI"""
        filename = self.filename.get()
        vid = cv2.VideoCapture(filename) # input video

        # spinlab logo to display in upper right corner of output video
        spinlab = cv2.imread('util/SpinLabUCLA_BW_strokes.png')
        # read the width and height of input video.
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = fps_var.get()
        fileName = savefile_var.get()
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v') # codec for output video
        video_writer = cv2.VideoWriter(fileName+'.avi', fourcc, fps, (width, height)) # VideoWriter object for editing and saving the output video

        dim = (int(0.2*width),int((0.2*height)/3))
        spinlab = cv2.resize(spinlab, dim, interpolation = cv2.INTER_CUBIC) # resize spinlab logo based on input video dimensions

        natural_rpm = table_rpm_Var.get()
        cam_rpm = cam_rpm_var.get()
        digi_rpm = digi_rpm_var.get()
        tot_rpm = cam_rpm + digi_rpm
        totOmega = (tot_rpm *2*np.pi)/60
        dtheta = digi_rpm*(6/fps)
        addRot = digi_rpm != 0

        change_units = unit_var.get()
        unitType = unit_type_var.get()
        unitCount = unit_count_var.get()
        unitConv = 1 			# intialize to 1 in the case that no unit conversion is selected

        startFrame = fps*start_time_var.get()
        if startFrame == 0:
            startFrame = 1
        numFrames = int(fps*(end_time_var.get() - start_time_var.get()))
        cust_mask = custom_mask_var.get()
        track_ball = track_var.get()
        makePlots = plot_var.get()

        # Close GUI window so rest of program can run
        root.destroy()

        # Open first frame from video, user will click on center
        vid.set(cv2.CAP_PROP_POS_FRAMES, startFrame) # set the first frame to correspond to the user-selected start time
        ret, frame = vid.read() # read the first frame from the input video
        frame = cv2.resize(frame,(width,height), interpolation = cv2.INTER_CUBIC) # ensure frame is correct dimensions
        cv2.namedWindow('CenterClick')

        # Use the appropriate mask-selecting function (depending on whether custom-shaped mask is checked)
        if cust_mask:
            cv2.setMouseCallback('CenterClick', nGon)
        else:
            cv2.setMouseCallback('CenterClick', circumferencePoints)

        # Append instructions to screen
        instructsCenter(frame)
        orig = frame.copy()
        while(1):
            cv2.imshow('CenterClick', frame)
            k = cv2.waitKey(0)
            if k == 13: # user presses ENTER
                break
            elif k == 127: # user presses BACKSPACE/DELETE
                removePoint(orig)

        cv2.destroyWindow('CenterClick')

        # Select initial position of ball (only if particle tracking is selected)
        if track_ball:
            vid.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
            ret, frame = vid.read()
            frame = cv2.resize(frame,(width,height), interpolation = cv2.INTER_CUBIC)
            cv2.namedWindow('Locate Ball')
            cv2.setMouseCallback('Locate Ball', locate)

            instructs_ball(frame)
            cv2.imshow('Locate Ball', frame)
            cv2.waitKey(0)
            cv2.destroyWindow('Locate Ball')

        # Draw a line to calculate a pixel-to-distance conversion factor.
        if change_units:
            vid.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
            ret, frame = vid.read()
            frame = cv2.resize(frame, (width, height),
                               interpolation=cv2.INTER_CUBIC)
            cv2.namedWindow('Distance Calibration')
            cv2.setMouseCallback('Distance Calibration', unitConversion)

            instructsUnit(frame)
            cv2.imshow('Distance Calibration', frame)
            cv2.waitKey(0)
            cv2.destroyWindow('Distance Calibration')

        # Reset video to first frame.
        vid.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

        # allocate empty arrays for particle-tracking data
        t = np.empty(numFrames)
        ball_x = np.empty(numFrames)
        ball_y = np.empty(numFrames)
        if track_ball:
            ballPts = 0 #will identify the number of times that Hough Circle transform identifies the ball
            lastLoc = particleCenter # most recent location of particle, initialized to the location the user selected
            thresh = 50
            tracking_data = np.zeros(numFrames) # logical vector which tells us if the ball was tracked at each frame
        framesArray = np.empty((numFrames,height,width,3), np.uint8)

        # np.savetxt('check.txt',vid.read()[1])

        # Go through the input movie frame by frame and do the following: (1) digitally rotate, (2) apply mask, (3) center the image about the point of rotation, (4) search for particle and record tracking results
        for i in range(numFrames):
            # Read + resize current frame
            ret, frame = vid.read() # read next frame from video
            frame = cv2.resize(frame,(width,height), interpolation = cv2.INTER_CUBIC)

            # (1) and (2) (the order they are applied depends on whether the movie is derotated)
            if tot_rpm != 0: # Case 1: the mask is applied before rotation so it co-rotates with the additional digital rotation
                cv2.fillPoly(frame, np.array([poly1, poly2]), 0) # black out everything outside the region of interest
                cv2.circle(frame, center, 4, (255,0,0), -1) # place a circle at the center of rotation

            if addRot:
                M = cv2.getRotationMatrix2D(center, i*dtheta, 1.0)
                frame = cv2.warpAffine(frame, M, (width, height))
                if tot_rpm == 0: # Case 2: the movie is de-rotated, we want to apply the mask after digital rotation so it is stationary
                    cv2.fillPoly(frame, np.array([poly1, poly2]), 0)
            else:
                cv2.fillPoly(frame, np.array([poly1, poly2]), 0)

            # (3)
            frame = centerImg(frame, center[0], center[1]) # center the image


            centered = cv2.resize(frame,(width,height), interpolation = cv2.INTER_CUBIC) # ensure the frame is the correct dimensions

            # (4)
            if track_ball: # if tracking is turned on, apply tracking algorithm
                gray = cv2.cvtColor(centered, cv2.COLOR_BGR2GRA_y) # convert to black and whitee
                gray = cv2.medianBlur(gray,5) # blur image. this allows for better identification of circles
                ballLoc = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=10, minRadius = int(particleRadius * 0.6), maxRadius = int(particleRadius * 1.4))
                if type(ballLoc) != type(None): # if a circle is identified, record it
                    for j in ballLoc[0,:]:
                        if (np.abs(j[0] - lastLoc[0]) < thresh) or (np.abs(j[1] - lastLoc[1]) < thresh):
                            cv2.circle(centered, (j[0],j[1]), j[2], (0,255,0),1)
                            cv2.circle(centered, (j[0],j[1]), 2, (0,0,255), -1)
                            ball_x[ballPts] = j[0]
                            ball_y[ballPts] = j[1]
                            t[ballPts] = i/fps
                            lastLoc = np.array([j[0],j[1]])
                            ballPts += 1
                            tracking_data[i] = 1
                            break

                # mark the frame with dots to indicate the particle path
                for k in range(ballPts-1):
                    cv2.circle(centered, (int(ball_x[k]), int(ball_y[k])), 1, (255,0,0), -1)


            annotateImg(centered, i) # apply diagnostic information and logos to each frame
            framesArray[i] = centered # save each frame in an array so that we can loop back through later and add the inertial radius

        # Done looping through video

        # Reformat tracking data
        if track_ball:
            ball_x = ball_x[0:ballPts] # shorten the array to only the part which contains tracking info
            ball_y = ball_y[0:ballPts]
            t = t[0:ballPts]
            ball_x -= center[0] # set the center of rotation as the origin
            ball_y -= center[1] # "                                      "

            # Convert from pixels to units of distance
            ball_x *= unitConv
            ball_y *= unitConv

            ballR = ((ball_x**2)+(ball_y**2))**(0.5) # convert to polar coordinates
            ball_theta = np.arctan2(ball_y, ball_x)   # "                          "
            for i in range(len(ball_theta)):        # ensure that the azimuthal coordinate is in the range [0, 2*pi]
                if ball_theta[i] < 0:
                    ball_theta[i] += 2*np.pi

            # Calculate velocities
            ux = calcDeriv(ball_x, t)
            uy = calcDeriv(ball_y, t)
            ur = calcDeriv(ballR, t)
            utheta = calcDeriv(ball_theta, t)
            utot = ((ux**2)+(uy**2))**(0.5)

            # Theoretical inertial frequency.
            fTh = 2*totOmega
            # Inertial radius, a combination of theory (fTh) and data (utot).
            r_inert = utot / fTh
            # Polynomial fit of degree 20 (provides a smooth fit through the
            # data and solves the uneven sampling problem).
            r_inert_smooth = splineFit(t, r_inert, 20)
            ux_smooth = splineFit(t, ux, 20)
            uy_smooth = splineFit(t, uy, 20)


            # If option to make plots of tracking data is selected, make plots
            if makePlots:
                plt.figure(1)
                plt.subplot(211)
                plt.plot(t, ballR, 'r1')
                plt.xlabel(r"$t$ (s)")
                plt.ylabel(r"$r$")

                plt.subplot(212)
                plt.plot(t, ball_theta, 'r1')
                plt.xlabel(r"$t$ (s)")
                plt.ylabel(r"$\theta$")
                plt.savefig(fileName+'_polar.pdf', format = 'pdf', dpi = 1200)

                plt.figure(2)
                plt.subplot(211)
                plt.plot(t, ball_x, 'r1')
                plt.xlabel(r"$t$ (s)")
                plt.ylabel(r"$x$")
                plt.subplot(212)
                plt.plot(t, ball_y, 'b1')
                plt.xlabel(r"$t$ (s)")
                plt.ylabel(r"$y$")
                plt.savefig(fileName+'_cartesian.pdf', format='pdf', dpi=1200)

                plt.figure(3)
                plt.subplot(221)
                plt.plot(t, ux, label=r"$u_x$")
                plt.plot(t, uy, label=r"$u_y$")
                plt.xlabel(r"$t$ (s)")
                plt.legend(loc='upper right', fontsize='x-small')
                plt.subplot(222)
                plt.plot(t, r_inert)
                plt.plot(t, r_inert_smooth)
                plt.xlabel(r"$t$ (s)")
                plt.ylabel(r"$r_i$")
                plt.subplot(223)
                plt.plot(t, ur)
                plt.xlabel(r"$t$ (s)")
                plt.ylabel(r"$u_r$")
                plt.subplot(224)
                plt.plot(t, utheta)
                plt.xlabel(r"$t$ (s)")
                plt.ylabel(r"$u_{\theta}$")
                plt.ylim([-3*np.abs(totOmega), 3*np.abs(totOmega)])
                plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5)
                plt.savefig(fileName+'_derivs.pdf', format='pdf', dpi=1200)

            # Record tracking data in a .txt file
            dataList = np.array([t, ball_x, ball_y, ballR, ball_theta, ux, uy,
                                 ur, utheta, utot])

            dataFile = open(fileName+'_data.txt', 'w')
            dataFile.write('DigiPyRo Run Details \n \n')
            dataFile.write('Original File: ' + filename + '\n' +
                           'Output File: ' + fileName + '\n')
            dataFile.write('Date of run: ' + time.strftime("%c") + '\n \n')
            dataFile.write('Original rotation of camera: ' + str(cam_rpm) +
                           ' RPM\n' + 'Added digital rotation: ' +
                           str(digi_rpm) + ' RPM\n' +
                           'Curvature of surface: ' + str(natural_rpm) +
                           ' RPM\n' + '\n' + 'Particle Tracking Data')
            dataFile.write(' in ' + unitType + ' and ' + unitType + '/s\n')
            dataFile.write('t x y r theta u_x u_y u_r u_theta ||u||\n')

            for i in range(len(ball_x)):
                for j in range(len(dataList)):
                    entry = "%.2f" % dataList[j][i]
                    if j < len(dataList) - 1:
                        dataFile.write(entry + ' ')
                    else:
                        dataFile.write(entry + '\n')
            dataFile.close()

        cv2.destroyAllWindows()
        vid.release()

        # Loop through video and report inertial radius.
        num_radii = 25
        if track_ball:
            r_inert_smooth[0:3] = 0
            # The first and last few inertial radii tend to have very large
            # systematic errors.  Set them to 0 so that they are not shown.
            r_inert_smooth[ballPts-3:ballPts] = 0
        # Only do this if particle tracking is selected and the inertial radius
        # is not infinite (happens when tot_rpm = 0).
        if track_ball and tot_rpm != 0:
            index=0
            line_start_x = np.empty(ballPts, dtype=np.int16)
            line_start_y = np.empty(ballPts, dtype=np.int16)
            line_end_x = np.empty(ballPts, dtype=np.int16)
            line_end_y = np.empty(ballPts, dtype=np.int16)
            for i in range(numFrames):
                frame = framesArray[i]
                if tracking_data[i]:
                    (line_start_x[index], line_start_y[index]) = (int(0.5+ball_x[index]+center[0]), int(0.5+ball_y[index]+center[1]))
                    angle = np.arctan2(uy_smooth[index],ux_smooth[index])
                    rad = r_inert_smooth[index]
                    (line_end_x[index], line_end_y[index]) = (int(0.5+center[0]+ball_x[index]+(rad*np.sin(angle))), int(0.5+center[1]+ball_y[index]-(rad*np.cos(angle))))
                    if index < num_radii:
                        numLines = index
                    else:
                        numLines = num_radii
                    for j in range(numLines):
                        cv2.line(frame, (line_start_x[index-j], line_start_y[index-j]), (line_end_x[index-j], line_end_y[index-j]), (int(255), int(255), int(255)), 1)
                    index+=1
                video_writer.write(frame)
        else:
            for i in range(numFrames):
                frame = framesArray[i]
                video_writer.write(frame)

        video_writer.release()

        side_by_side_view = side_by_side_view_var.get()
        if side_by_side_view:
            oldVid = cv2.VideoCapture(filename)
            newVid = cv2.VideoCapture(fileName+'.avi')
            # Reset original video to start frame.
            oldVid.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
            newVid.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

            border_width = 30
            border_height = 100
            new_width = int(4*r) + border_width
            new_height = int(2*r) + 2*border_height

            old_height1 = (int((height/2)+r))
            old_height2 = (int((height/2)-r))
            old_width1 = (int((width/2)+r))
            old_width2 = (int((width/2)-r))
            if old_height1 > height:
                old_height1 = height
            if old_width1 > width:
                old_width1 = width
            if old_height2 < 0:
                old_height2 = 0
            if old_width2 < 0:
                old_width2 = 0

            if ((old_height1-old_height2) - (2*border_height)) < height:
                new_height = (old_height1-old_height2) + 2*border_height
            if ((old_width1-old_width2) - (2*border_width)) < width:
                new_width = 2*(old_width1-old_width2) + border_width

            video_writerSBS = cv2.VideoWriter(fileName+'SideBySide'+'.avi',
                                              fourcc, fps, (new_width,
                                              new_height))
            for i in range(numFrames):
                # Grab frames from original and DigiPyRo-ed movie, resize them and then put them side by side
                ret1, frame1 = oldVid.read()
                ret2, frame2 = newVid.read()
                if frame1 is None or frame2 is None:
                    continue
                cv2.fillPoly(frame1, np.array([poly1,poly2]),0) # apply mask to original movie
                frame1 = centerImg(frame1, center[0], center[1]) # center original movie about rotation point

                outFrame = np.zeros((new_height,new_width,3), np.uint8)
                outFrame[border_height:new_height-border_height,0:(old_width1-old_width2)] = frame1[old_height2:old_height1, old_width2:old_width1]
                outFrame[border_height:new_height-border_height,int(2*r)+border_width:new_width] = frame2[old_height2:old_height1, old_width2:old_width1]
                annotateSBS(outFrame)

                video_writerSBS.write(outFrame)
            video_writerSBS.release()

    def create_gui_menu(self, title="digipyro", start_text="Start!"):
        """Create the GUI menu."""
        root = Tk()
        root.title(title)
        start_button = Button(root, text=start_text, command=start)
        start_button.grid(row=11, column=0)

        table_rpm_var = DoubleVar()
        table_rpm_entry = Entry(root, textvariable=table_rpm_var)
        table_text = "Curvature of table (in RPM, enter 0 for a flat surface):"
        table_label = Label(root, text=table_text)
        table_rpm_entry.grid(row=0, column=1)
        table_label.grid(row=0, column=0)

        digi_rpm_var = DoubleVar()
        cam_rpm_var = DoubleVar()
        digi_rpm_entry = Entry(root, textvariable=digi_rpm_var)
        cam_rpm_entry = Entry(root, textvariable=cam_rpm_var)
        digi_label = Label(root, text="Additional digital rotation (RPM):")
        cam_label = Label(root, text="Physical rotation (of camera, RPM):")
        digi_rpm_entry.grid(row=2, column=1)
        cam_rpm_entry.grid(row=1, column=1)
        digi_label.grid(row=2, column=0)
        cam_label.grid(row=1, column=0)

        side_by_side_view_var = BooleanVar()
        side_by_side_view_entry = Checkbutton(root, text="Display original video side-by-side with DigiPyRo-ed video", variable = side_by_side_view_var)
        side_by_side_view_entry.grid(row=2, column=2)

        custom_mask_var = BooleanVar()
        custom_mask_entry = Checkbutton(root, text="Custom-Shaped Mask (checking this box allows for a polygon-shaped mask. default is circular)", variable=custom_mask_var)
        custom_mask_entry.grid(row=3, column=0)

        filename_var = StringVar()
        filename_entry = Entry(root, textvariable=filename_var)
        filename_label = Label(root, text="Full filepath to movie:")
        filename_entry.grid(row=4, column=1)
        filename_label.grid(row=4, column=0)

        savefile_var = StringVar()
        savefile_entry = Entry(root, textvariable=savefile_var)
        savefile_label = Label(root, text="Save output video as:")
        savefile_entry.grid(row=5, column=1)
        savefile_label.grid(row=5, column=0)

        start_time_var = DoubleVar()
        end_time_var = DoubleVar()
        start_time_entry = Entry(root, textvariable=start_time_var)
        end_time_entry = Entry(root, textvariable=end_time_var)
        start_time_label = Label(root, text="Start and end times (in seconds):")
        start_time_label.grid(row=6, column=0)
        start_time_entry.grid(row=6, column=1)
        end_time_entry.grid(row=6, column=2)

        track_var = BooleanVar()
        track_entry = Checkbutton(root, text="Track Ball", variable=track_var)
        track_entry.grid(row=5, column=2)
        plot_var = BooleanVar()
        plot_entry = Checkbutton(root, text="Create plots with tracking results",
                                 variable=plot_var)`
        plot_entry.grid(row=5, column=3)

        fps_var = DoubleVar()
        fps_entry = Entry(root, textvariable=fps_var)
        fps_label = Label(root, text="Frames per second of video:")
        fps_entry.grid(row=7, column=1)
        fps_label.grid(row=7, column=0)

        unit_var = BooleanVar()
        unit_entry = Checkbutton(root, text="Add distance units calibration",
                                 variable=unit_var)
        unit_entry.grid(row=8, column=0)
        unit_type_var = StringVar()
        unit_type_entry = Entry(root, textvariable=unit_type_var)
        unit_type_label = Label(root, text="Length unit (e.g. cm, ft):")
        unit_count_var = DoubleVar()
        unit_count_label = Label(root, text="Unit count:")
        unit_count_entry = Entry(root, textvariable=unit_count_var)
        unit_type_label.grid(row=8, column=1)
        unit_type_entry.grid(row=8, column=2)
        unit_count_label.grid(row=8, column=3)
        unit_count_entry.grid(row=8, column=4)

        root.mainloop()


if __name__ == "__main__":
    create_gui_menu()
