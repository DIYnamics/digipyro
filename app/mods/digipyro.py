import argparse

import cv2
import numpy as np

import interaction as interact
#------------------------------------------------------------------------------
# *** COMMAND LINE INTERFACE SETUP ***
# initial message for program
msg = """ This program digitally rotates a movie. It can take a video from
the synthetic program as well as other movies to rotate and place the user in
the rotating frame. The user may input several variables and will need to
select an area of interest, so the program rotates about the correct axis."""
fmt = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description=msg,
                                 formatter_class=fmt)

# collecting arguments for the user to change
parser.add_argument('--t0', type=float, default=0,
                    help='The start time for the digital rotation.')

parser.add_argument('--t1', type=float, default=0,
                    help=('The end time for the digital rotation. ',
                          'If left to default, it will go through the entire ',
                          'film.'))

parser.add_argument('--rpm', type=float, default=10,
                   help=('The digital rotation given to the movie. This ',
                         'rotation will be subtracted from the rotation of ',
                         'the film and is consistent with a right-handed ',
                         'frame of reference.'))

parser.add_argument('--path', type=str, required=True,
                    help=('The complete path of the movie being inputted.',
                          'The output will be saved in the same directory of ',
                          'where this movie is located.'))
#------------------------------------------------------------------------------

# collecting user input into list
args = parser.parse_args()

t0 = args.t0
t1 = args.t1
rpm = args.rpm
path = args.path

def digi_rotate(t0, t1, rpm, path):
    """Digitally rotates a movie.

    Parameters
    ----------
    t0 : float
        The initial start time of the movie [s].
    t1 : float
        The final end time of the movie [s].
    rpm : float
        The rotation the film [rotations per min].
    path : str
        The path to the movie. The full file name and extension should
        be given so the program executes correctly.

    Notes
    -----
    At the end of execution a new movie will be saved in the same directory
    as the movie given. It will also be appended with a '-rot' to the file
    name to show that this is the rotated version of the movie.
    """
    # film given
    vid = cv2.VideoCapture(path)

    # collecting frame values from film
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dim = (width, height)

    # find the starting frame position
    start = int(fps * t0)

    # if default, variable becomes the entire length of the film
    # else it becomes the length the user desires
    if t1 == 0:
        # error gets thrown if you go the complete end, so I stop 5
        # frames before
        frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 5
    else:
        frames = int(fps * (t1 - t0))

    # remove extension of film
    path = path[:-4]
    # create new file name
    output = path + '-rot.mp4'
    # codecc and new film to be outputted
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(output, fourcc, fps, dim)

    # poly are used to blackout area outside of selection
    # center is used as the axis of rotation
    poly1, poly2, center = interact.selection_window(vid, dim, start)

    # each rotation step
    # the negative is for the right hand rule
    # -1 * (360 deg / 1 rot) * (1 min / 60 sec) * rpm * fps
    dtheta = -1 * 6 * rpm / fps

    # performing rotation
    for i in range(frames):
        # collect current frame and resize
        ret, frame = vid.read()
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)

        # blackout region outside
        cv2.fillPoly(frame, np.array([poly1, poly2]), 0)
        cv2.circle(frame, center, 4, (255,0,0), -1)

        # rotate frame
        M = cv2.getRotationMatrix2D(center, i*dtheta, 1.0)
        frame = cv2.warpAffine(frame, M, dim)

        # re-center and re-size frame
        frame = interact.center_frame(frame, center[0], center[1], dim)
        centered = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)
        # write new frame to output
        video_writer.write(centered)

    # save output
    video_writer.release()

# module runs on cLI if run on its own
if __name__ == "__main__":
    digi_rotate(t0, t1, rpm, path)
