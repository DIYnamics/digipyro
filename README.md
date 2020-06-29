![](https://github.com/DJ-2805/DigiPyRo/blob/master/util/SpinLabUCLA_BW_strokes.png)

DigiPyRo is a program designed to digitally rotate a movie and allows
for single-particle tracking. This was originally designed to
intuitively show Coriolis force effects through the appearance of
inertial circles when digitally rotating film of a ball oscillating on a
parabolic surface.

The Python 2.7 version of the code was created by [Sam
May](https://github.com/sam-may/DigiPyRo) and the current version is
maintained by [David James](https://github.com/DJ-2805) in conjunction
with the [SPIN Lab](https://spinlab.epss.ucla.edu/) at UCLA.

Setup
=====

You'll first need to get the files needed to install and run DigiPyRo.

1.  First download the files needed. You can either follow
    [this](https://github.com/DJ-2805/DigiPyRo) and download the files
    or if you have `git` on your terminal you can download through
    terminal.

    ``` {.bash}
    git clone https://github.com/dj-2805/digipyro
    ```

    -   If you instead followed the link, then click on the green
        `Clone or download` button, and download the repository into
        your machine

Installation
============

This SOP assumes that you have python already installed. Before you get
to run DigiPyRo you'll need to install the dependencies so the program
can run. Because of how most of this program is setup, you will do most
of the interaction with code through the terminal.

1.  If you are on Windows, open the `Powershell`. If you are on Mac,
    open the `Terminal`.
2.  To verify that you do have Python, run the following command

    ``` {.bash}
    # windows machine
    python --version

    # MAC machine
    python3 --version
    ```

    -   you should see an output of Python 3.x.x version number. If you
        see Python 2.x.x, you will need to reinstall your Python.

    -   NOTE :: Some machines need `python3` being called explicitly
        rather than `python` in terminal calls. If this produces the
        correct version, then any future commands you see in the rest of
        the instructions will require `python3` being typed rather than
        `python`

3.  First make sure your package manager is up-to-date.

    ``` {.bash}
    # if you are on a Windows machine, run this command
    python -m pip install --upgrade pip

    # if you are on a Mac, run this command
    pip install --upgrade pip
    ```

4.  Next you will need to move over to the the directory that contains
    the files that you unzipped.
5.  Now you'll be installing the dependencies.

    ``` {.bash}
    pip install -r requirements.txt
    ```

Synthetic Movie Program
=======================

At this point, you should have Python and all the dependencies to run
the programs. This first program creates a synthetic .avi movie for use
with DigiPyRo. If you already have a film from experimentation, then you
can skip this program, and move to the DigiPyRo program below.

The video shows a ball rolling on a parabolic surface, where the user
may change the video length of the movie, frame rate, resolution,
frequency of oscillations, rotation rate of the reference frame, and
control the inital conditions of the ball.

1.  You will still need the terminal at this point, so open it up if you
    have closed it.
2.  To see the `help` message for the the movie program type

    ``` {.bash}
    python synth.py -h
    # or
    python synth.py --help
    ```

3.  You should see the following messaged

    ``` {.org}
    usage: synth.py [-h] [-t TIME] [-f FPS] [-w WIDTH] [-l LENGTH] [-r EQPOT_RPM] [-R CAM_RPM] [--r0 R0] [--vr0 VR0] [--phi0 PHI0] [--vphi0 VPHI0]

    This program creates a synthetic .avi movie for use with DigiPyRo. The video shows a ball rolling on a parabolic surface. The user may change the length of the movie,
    the frame rate, the resolution of the movie, the frequency of oscillations, the rotation rate of the reference frame, and control the initial conditions of the ball.

    optional arguments:
      -h, --help            show this help message and exit
      -t TIME, --time TIME  The desired length of the movie in seconds. (default: 5)
      -f FPS, --fps FPS     The frame rate of the video (frames per second). Set this to a low value (10-15) for increased speed or a higher value (30-60) for better
                            results. (default: 30.0)
      -w WIDTH, --width WIDTH
                            The width in pixels of the video. (default: 1260)
      -l LENGTH, --length LENGTH
                            The height in pixels of the video. (default: 720)
      -r EQPOT_RPM, --eqpot_rpm EQPOT_RPM
                            The deflection of a stationary paraboloid surface as if it were an equipotentional in a system rotating at the specified rate. A good value
                            would be between 5 and 15. (default: 10.0)
      -R CAM_RPM, --cam_rpm CAM_RPM
                            The rotation rate of the camera. The two natural frames of reference are with rotRate = 0 and rotRate = rpm. (default: 0.0)
      --r0 R0               The initial radial position of the ball. Choose a value betweeon 0 and 1. (default: 1.0)
      --vr0 VR0             The initial radial velocity of the ball. A good value would be between 0 and 1. (default: 0.0)
      --phi0 PHI0           The initial azimuthal position of the ball. Choose a value between 0 and 2*pi. (default: 0.7853981633974483)
      --vphi0 VPHI0         The initial azimuthal velocity of the ball. A good value would be between 0 and 1. (default: 0)
    ```

4.  The program has several flags, and all of them have default values
    described in the help message. The program can be run in several
    different ways, where you run it with it's defualt values or you
    change the flags desired to have a video produced. Note, when you do
    run it, you'll be prompted for a movie name. The produced movie will
    be in `.avi` format, so only a movie name will be needed. Examples
    are shown below:

    ``` {.bash}
    # running the program with just its default values
    python synth.py

    # running the program with changing one value
    python synth.py -t 10

    # the same but using the full flag name
    python synth.py --time 10

    # running the program with multiple flags
    python synth.py -t 7 -r 15 --vr0 3
    ```

    NOTE
    :   Depending on the parameters given and how powerful your machine
        is, this could take a couple minutes for it to create your film.

5.  After the program completes, you should see your movie file created
    in the same directory.

DigiPyRo Program
================

Now to run the DigiPyRo program you'll need a video and to set some
parameters. The DigiPyRo program runs through a GUI, but will still need
a terminal command to get the program started.

1.  Run the following command to get DigiPyRo started.

    ``` {.bash}
    python DigiPyRo.py
    ```

2.  You will see a GUI window appear, where you can input values for
    each area.
3.  The values are based on either the video you created from synth.py
    or a lab experiment you ran beforehand.
4.  For the `full filepath to movie` parameter, unless the video is in
    the same directory as `DigiPyRo.py`, you must specify the entire
    path. Furthermore, you must specify the extension of the movie (i.e.
    .avi, .mp4, etc.).
5.  For the `Save output video as` parameter, only the file name needs
    to be given. The extension will be added after the program executes.
6.  For the `Start and end times` parameter, the end time should be a
    bit shorter of the true end of the film (i.e. if the film is 5 secs
    long then this `end time` should go to 4.8). This is a current bug
    that I came across; sometimes the program goes out of index if the
    full length of the film is given.
7.  For more description and instructions on the programs refer to
    [Sam's Instruction
    PDF](https://github.com/DJ-2805/DigiPyRo/blob/master/Examples/BasicExamples_v3.pdf).

    NOTE
    :   Sam's PDF is out-of-date for some instructions, because the
        program has been changed, but still gives description and images
        on some of the steps.


