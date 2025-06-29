#!/usr/bin/env python3
"""Initial Exponential"""


class Exponential:
    """Exponential that represents an exponential distribution"""
    e = 2.7182818285
    π = 3.1415926536

    def __init__(self, data=None, lambtha=1.):
        """Class constructor

        data: list of the data to be used to estimate the distribution
        lambtha: the expected number of occurrences in a given time frame"""
        if data is None:
            self.lambtha = float(lambtha)
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """Function that calculates the value of the PDF for a given time
        period

        x: is the time period

        Return: PDF value of x"""
        if x < 0:
            return 0
        return self.lambtha * (self.e ** (-self.lambtha * x))

    def cdf(self, x):
        """Function that calculates the value of the CDF for a given time
        period

        x: is the time period

        Return: CDF value for x"""
        if x < 0:
            return 0
        return 1 - self.e ** (-self.lambtha * x)
