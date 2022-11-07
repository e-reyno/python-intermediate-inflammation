"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains
inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array for each day.

    :param data: 2d array with inflammation data
    :returns: array of mean values of measurements for each day
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: 2d array with inflammation data
    :returns: array of maximum values of measurements for that day
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array.

    :param data: 2d array with inflammation data
    :returns: array of minimum values of measurements for each day
    """
    return np.min(data, axis=0)

def patient_normalise(data):
    '''Normalise patient data from 2D array of inflammation data'''
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')
    maxes = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / maxes[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised


def daily_standard_deviation(data):
    """Calculate the patient standard deviation  of a 2d inflammation data array
    """
    return np.std(data, axis=0)