import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from PIL import Image

import paraboloid as pbd

# sets plots to a darker color scheme
plt.style.use("dark_background")
#------------------------------------------------------------------------------
# *** COMMAND LINE INTERFACE SETUP ***
# initial message for program
msg = """ This program creates a synthetic .mp4 movie for use with DigiPyRo.
The video shows a puck sliding on a parabolic surface. The user may change
several variables of the puck, and observe the animation in the output."""
fmt = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description = msg,
                                 formatter_class=fmt)

# collecting arguments for the user to change
parser.add_argument('-t', '--time', type=float, default=6,
                    help='The length of the movie in seconds.')

parser.add_argument('-r', '--rpm_topo', type=float, default=10,
                    help=('The deflection of a stationary paraboloid surface '
                          'as if it were an equipotential in a system '
                          'rotating at the specified RPM. '
                          'A good value would be between 5 and 15.'))

parser.add_argument('-u', '--u0', type=float, default=0,
                    help=('The initial x-velocity in cm/s of the puck moving '
                          'on the parabolic surface. If too high, the puck '
                          'will go off the edge during the animation.'))

parser.add_argument('-v', '--v0', type=float, default=0,
                    help=('The initial y-velocity in cm/s of the puck moving '
                          'on the parabolic surface. If too high, the puck '
                          'will go off the edge during the animation.'))

parser.add_argument('-x', '--x0', type=float, default=1,
                    help=('The initial x-position in cm of the puck moving on '
                          'the parabolic surface. If too high, the puck will '
                          'go off the edge during the animation.'))

parser.add_argument('-R', '--radius', type=float, default=2,
                    help='The radius of the paraboloid in cm.')

parser.add_argument('-n', '--name', type=str, default="movie",
                    help=("Title of the movie that will be outputted. "
                          "The extension does not need to be provided; the "
                          ".mp4 will be added in the program."))

parser.add_argument('-s', '--switch', type=int, default=0,
                    help=('Choice to either show the animation or save it. '
                          'Default displays the animation. Option [1] will '
                          'save the animation as an mp4.'))
#------------------------------------------------------------------------------

# collecting user input into list
args = parser.parse_args()

time = args.time
omega = 2*np.pi*args.rpm_topo/60
u0 = args.u0
v0 = args.v0
x0 = args.x0
radius = args.radius
name = args.name + ".mp4"
switch = args.switch

def animate_paraboloid(time, omega, u0, v0, x0, radius):
    """Animates the paraboloid.

    Parameters
    ----------
    time : float
        length of animation in seconds
    omega : float
        effective rotation [1/s]
    u0 : float
        initial x-component of the velocity [cm/s]
    v0 : float
        initial y-component of the velocity [cm/s]
    x0 : float
        initial x-component of the position [cm]

    Returns
    -------
    Obj, animation object
        reference matplotlib.animation.funcAnimation for more info.
    """

    # creating circle
    circle = pbd.circle(radius)
    parabola = pbd.parabola(radius, omega)

    # size of plots based off paraboloid radius
    rmax = radius*1.5
    size = (-1*(rmax), rmax)

    xt, yt = [], []

    resize = (150,75)
    spinlab = Image.open('../static/SpinLabUCLA_BW_strokes.png')
    spinlab = spinlab.resize(resize)
    diynamics = Image.open('../static/diynamics-swirl-light.png')
    diynamics = diynamics.resize(resize)

    # creating figure and plots
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6,8),
                                 gridspec_kw={'height_ratios': [3,1]})

    # inserting watermark
    fig.figimage(diynamics, xo=1000)
    fig.figimage(spinlab, xo=10)

    # a0 plots out the top-view of the paraboloid
    a0.set_xlim(size)
    a0.set_ylim(size)
    a0.set_title("Top-View")
    a0.set_xlabel("X [cm]")
    a0.set_ylabel("Y [cm]")

    a0.add_patch(circle)
    # follows the puck path in the absolute axis
    puckTop, = a0.plot([], [], linestyle='none',
                       marker='o', mfc='white', mec='red', label="Puck")
    # follows the puck path in the rotating axis
    puckInt, = a0.plot([], [], linestyle='none', marker='.',
                       mfc='green', ms=1, label="Inertial Path")
    a0.legend()

    # a1 plots out the side-view of the paraboloid
    a1.set_xlim(size)
    a1.set_xlabel("X [cm]")
    a1.set_ylabel("Z [cm]")
    a1.set_title("Side-View")

    a1.plot(parabola[0], parabola[1], color='white', label="Paraboloid")
    # follows side view of the puck path
    puckSide, = a1.plot([], [], linestyle='none',
                        marker='o', mfc='white', mec='red', label="Puck")

    plt.tight_layout()

    # calculating out the values for each from
    fps = 30
    t = np.linspace(start=0, stop=time, num=int(time*fps))
    frames = len(t)
    x, y, z = pbd.position(t, omega, u0, v0, x0)
    dot = omega * t[1]

    def init():
        """Initialization function for the animation.

        """
        return puckTop, puckInt, puckSide

    def animation_frame(i):
        """Specific frame of that animation.

        """
        xpos = x[i]
        ypos = y[i]
        zpos = z[i]

        # rotation of trajectory inprint on rotating axis
        for j in range(i):
            xn = xt[j] * np.cos(dot) - yt[j] * np.sin(dot)
            yn = xt[j] * np.sin(dot) + yt[j] * np.cos(dot)
            xt[j] = xn
            yt[j] = yn

        # adding points to the rotating axis
        xt.append(xpos)
        yt.append(ypos)

        # setting points for animation
        puckTop.set_data(xpos, ypos)
        puckInt.set_data(xt, yt)
        puckSide.set_data(xpos, zpos)

        return puckTop, puckInt, puckSide

    animation = animate.FuncAnimation(fig,
                                      init_func=init,
                                      func=animation_frame,
                                      frames=frames,
                                      blit=True)
    return animation

def save_animation(animation, name):
    """Saves animation when called.

    """
    animation.save(name, writer='ffmpeg', fps=30, dpi=200)

# module runs on CLI if run on its own
if __name__ == "__main__":
    animation = animate_paraboloid(time, omega, u0, v0, x0, radius)

    if switch == 0:
        plt.show()
    elif switch == 1:
        save_animation(animation, '../../../' + name)