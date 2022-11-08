"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains
inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""
#init is the initialiser method - setting inital values. similar to constructor
#self refers to instance we are calling and gets filled in by python
#can add behaviours to classes - use methods - function inside class
import numpy as np

class Observation:
    def __init__(self, day, value):
        self.day = day 
        self.value = value

    def __str__(self):
        return str(self.value)


class Person:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        #overriding default string method so print self.name instead of object ref
        return self.name

class Patient(Person):
    
    def __init__(self, name: str):
        super().__init__(name)
        self.observations = []

    def __str__(self):
        #overriding default string method so print self.name instead of object ref
        return self.name

    #value of self set to this object
    def add_observation(self, value, day=None):
        """_summary_

        Args:
            value (_type_): _description_
            day (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if day is None:
            try:
                day = self.observations[-1]['day'] + 1
            except IndexError:
                day = 0
        new_observation = Observation(day, value)
        self.observations.append(new_observation)
        return new_observation

    @property
    def last_observation(self):
        return self.observations[-1]


class Doctor(Person):
    def __init__(self, name, patients):
        self.name = name
        self.patients = patients
    def __str__(self):
        return self.name

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

