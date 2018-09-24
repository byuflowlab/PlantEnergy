#!/usr/bin/env python
# encoding: utf-8
# this file contains methods for different approaches to sampling across a rotor plane

import numpy as np

def hermite_spline(x, x0, x1, y0, dy0, y1, dy1):
    #    This function produces the y and dy values for a hermite cubic spline
    #    interpolating between two end points with known slopes
    #
    #    :param x: x position of output y
    #    :param x0: x position of upwind endpoint of spline
    #    :param x1: x position of downwind endpoint of spline
    #    :param y0: y position of upwind endpoint of spline
    #    :param dy0: slope at upwind endpoint of spline
    #    :param y1: y position of downwind endpoint of spline
    #    :param dy1: slope at downwind endpoint of spline
    #
    #    :return: y: y value of spline at location x

    # initialize coefficients for parametric cubic spline
    c3 = (2.0*(y1))/(x0**3 - 3.0*x0**2*x1 + 3.0*x0*x1**2 - x1**3) - \
         (2.0*(y0))/(x0**3 - 3.0*x0**2*x1 + 3.0*x0*x1**2 - x1**3) + \
         (dy0)/(x0**2 - 2.0*x0*x1 + x1**2) + \
         (dy1)/(x0**2 - 2.0*x0*x1 + x1**2)

    c2 = (3.0*(y0)*(x0 + x1))/(x0**3 - 3.0*x0**2*x1 + 3.0*x0*x1**2 - x1**3) - \
         ((dy1)*(2.0*x0 + x1))/(x0**2 - 2.0*x0*x1 + x1**2) - ((dy0)*(x0 +
         2.0*x1))/(x0**2 - 2.0*x0*x1 + x1**2) - (3.0*(y1)*(x0 + x1))/(x0**3 -
         3.0*x0**2*x1 + 3.0*x0*x1**2 - x1**3)

    c1 = ((dy0)*(x1**2 + 2.0*x0*x1))/(x0**2 - 2.0*x0*x1 + x1**2) + ((dy1)*(x0**2 +
         2.0*x1*x0))/(x0**2 - 2.0*x0*x1 + x1**2) - (6.0*x0*x1*(y0))/(x0**3 -
         3.0*x0**2*x1 + 3.0*x0*x1**2 - x1**3) + (6.0*x0*x1*(y1))/(x0**3 -
         3.0*x0**2*x1 + 3.0*x0*x1**2 - x1**3)

    c0 = ((y0)*(- x1**3 + 3.0*x0*x1**2))/(x0**3 - 3.0*x0**2*x1 + 3.0*x0*x1**2 -
         x1**3) - ((y1)*(- x0**3 + 3.0*x1*x0**2))/(x0**3 - 3.0*x0**2*x1 +
         3.0*x0*x1**2 - x1**3) - (x0*x1**2*(dy0))/(x0**2 - 2.0*x0*x1 + x1**2) - \
         (x0**2*x1*(dy1))/(x0**2 - 2.0*x0*x1 + x1**2)

    # Solve for y and dy values at the given point
    y = c3*x**3 + c2*x**2 + c1*x + c0
    dy_dx = c3*3*x**2 + c2*2*x + c1

    return y, dy_dx

def sunflower_points(n, alpha=1.0):
    # this function generates n points within a circle in a sunflower seed pattern
    # the code is based on the example found at
    # https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle

    def radius(k, n, b):
        if (k + 1) > n - b:
            r = 1. # put on the boundary
        else:
            r = np.sqrt((k + 1.) - 1. / 2.) / np.sqrt(n - (b + 1.) / 2.)  # apply squareroot

        return r

    x = np.zeros(n)
    y = np.zeros(n)

    b = np.round(alpha * np.sqrt(n)) # number of boundary points

    phi = (np.sqrt(5.) + 1.) / 2.  # golden ratio

    for k in np.arange(0, n):

        r = radius(k, n, b)

        theta = 2. * np.pi * (k+1) / phi**2

        x[k] = r * np.cos(theta)
        y[k] = r * np.sin(theta)

    return x, y


def circumference_points(npts, location=0.735):

    alpha = 2.*np.pi/npts
    x = np.zeros(npts)
    y = np.zeros(npts)

    for n in np.arange(0, npts):
        y[n] = np.sin(alpha*n)*location
        x[n] = np.cos(alpha*n)*location

    return x, y


def line_points(npts):

    x = np.linspace(-1., 1., npts)
    y = np.zeros(npts)

    return x, y


# if __name__ == '__main__':