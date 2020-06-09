import numpy as np

g = 9.817

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
        x, y, and z componets of the position [cm]
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
        radius of the circle [cm]

    Returns
    -------
    xc, yc : array_like
        x and y values of the circle
    """
    twopi = 2 * np.pi

    k = np.arange(1, 102)
    xc = r * np.cos(k * twopi / 100)
    yc = r * np.sin(k * twopi / 100)

    return xc, yc

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
