import numpy as np
import matplotlib.patches as patch

g = 981.7

def position(t, omega, u0, v0, x0):
    """Finds position of puck on paraboloid. The equations come from
    Cushman-Roisin Chapter 2, in particular, 2.17a, and 2.17b.

    Parameters
    ----------
    t : float
        time of given moment [s]
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
    x, y, z : floats
        X, y, and z componets of the position [cm].
    """
    ot = omega * t

    x = x0 * np.cos(ot) + u0 * np.sin(ot) / omega
    y = v0 * np.sin(ot) / omega
    z = omega**2 / (2*g) * (x**2 + y**2)

    return x, y, z

def circle(r):
    """Creates circle that represents edge of the paraboloid.

    Parameters
    ----------
    r : float
        Radius of the circle [cm].

    Returns
    -------
    circle : object
        Circle object to be plotted.

    Notes
    -----
    Check matplotlib.patches.Circle for more information on the circle object.
    """
    circle = patch.Circle((0, 0), r, facecolor='none',
                          edgecolor='white',
                          linewidth=1.5)

    return circle

def parabola(r, omega):
    """Creates parabola that represents the sides of the paraboloid along y=0.

    Parameters
    ----------
    r : float
        radius of the paraboloid [cm]
    omega : float
        effective rotation [1/s]

    Returns:
    x, z : array_like
        x and z values of the parabola
    """

    x = np.linspace(start=-r,stop=r)
    z = omega**2 / (2 * g) * x**2

    return x, z

def check_edge(x, y, r):
    """Checks if puck is off the edge of the paraboloid.

    Parameters
    ----------
    x, y :  floats
        x and y positions of the puck [cm]
    r : float
        radius of the circle

    Returns
    -------
    bool
        True if puck crossed the radius of the paraboloid
    """

    if(r > np.sqrt(x**2 + y**2)):
        return True
    return False
