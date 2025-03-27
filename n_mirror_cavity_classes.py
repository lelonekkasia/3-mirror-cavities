import numpy as np 
import sympy as sp
from sympy import Symbol, symbols, Matrix
import matplotlib.pyplot as plt

t, r, L, delta, k = symbols("t r L delta k")

generic_mirror_matrix = sp.lambdify((r, t), (1 / (1j * t)) * Matrix([[-1, -r], [r, 1]]))
generic_propagation_matrix = sp.lambdify((L, delta, k) ,Matrix([[sp.exp(1j * k * (L + delta)), 0], [0, sp.exp(-1j * k * (L + delta))]]))

class Mirror:
    """Pass the big T"""
    def __init__(self, t):
        self.t = np.sqrt(t)
        self.r = np.sqrt(1 - self.t ** 2)
    
    def get_matrix(self):
        return generic_mirror_matrix(self.r, self.t)

    def __repr__(self): 
        return f"Transmissivity = {self.t}, Reflectivity = {self.r}"


class Propagation:
    def __init__(self, L, delta, wavelength):
        self.L = L
        self.delta = delta
        self.k =  (2 * np.pi) /  wavelength

    def get_matrix(self, subs_dict = None, lambdify_vars = None):
        return generic_propagation_matrix(self.L, self.delta, self.k)

class Subcavity:
    def __init__(self, L, delta, wavelength): 
        self.L = L
        self.delta = delta
        self.wavelength = wavelength
        self.k = (2 * np.pi) / wavelength 
        self.total_L = self.L + self.delta
        self.propagation = Propagation(self.L, self.delta, self.wavelength) 

    def get_matrix(self):
        return self.propagation.get_matrix()

    def __repr__(self):  
        return f"Subcavity(L={self.L}, delta={self.delta}, total_L={self.total_L})"



class Cavity:
    """
    Initializes a cavity with subcavities and mirrors.

    Parameters:
    - subcavity_params: List of (L, delta) tuples for each subcavity.
    - mirror_transmissivities: List of transmissivity values for mirrors separating subcavities.
    - k: Wave number (constant for all subcavities).
    """
    def __init__(self, subcavity_params, mirror_transmissivities, wavelength): 
        if len(mirror_transmissivities) != len(subcavity_params) + 1:
            raise ValueError("Number of mirrors should be one more than the number of subcavities.")

        # ennumerates subcavities starting with 1 !!!!
        self.wavelength = wavelength
        self.k =  (2 * np.pi) / wavelength
        self.subcavities = {
            f"subcavity_{i+1}": Subcavity(L, delta, wavelength) 
            for i, (L, delta) in enumerate(subcavity_params)
        }
        self.mirrors = {
            f"mirror_{i+1}": Mirror(t) 
            for i, t in enumerate(mirror_transmissivities)
        }
        self.total_length = sum(subcavity.total_L for subcavity in self.subcavities.values())

    def get_subcavity(self, index):
        """Retrieve a specific subcavity by its index (starting from 1)"""
        return self.subcavities.get(f"subcavity_{index}", None)

    def get_mirror(self, index):
        """Retrieve a specific mirror by its index (starting from 1)"""
        return self.mirrors.get(f"mirror_{index}", None)

    def get_propagation_matrix(self, index):
        """Retrieve the propagation matrix of a specific subcavity"""
        subcavity = self.get_subcavity(index)
        return subcavity.get_matrix() if subcavity else None

    def get_mirror_matrix(self, index):
        """Retrieve the mirror matrix of a specific mirror"""
        mirror = self.get_mirror(index)
        return mirror.get_matrix() if mirror else None

    def transfer_matrix(self):
        """Compute the total transfer matrix"""
        total_matrix = np.eye(2)
        matrices = [self.get_mirror_matrix(len(self.mirrors))]
        for i in range(len(self.subcavities), 0, -1):  # From N to 1
            matrices.append(self.get_propagation_matrix(i))  # Add propagation matrix
            matrices.append(self.get_mirror_matrix(i))  # Add mirror matrix
        
        for matrix in matrices:
            total_matrix = np.matmul(total_matrix, matrix)

        return total_matrix

    def transmission_coefficient(self):
        """Compute the transmission coefficient"""
        return -self.transfer_matrix()[1, 0] / self.transfer_matrix()[1, 1]

    def reflection_coefficient(self):
        """Compute the reflection coefficient"""
        return (self.transfer_matrix()[0, 0]*self.transfer_matrix()[1, 1] - self.transfer_matrix()[0, 1]*self.transfer_matrix()[1, 0]) / self.transfer_matrix()[1, 1]

    def transmitted_power(self):
        """Plot the transmitted power as a function of detuning"""
        powers = []
        for wavelength_value in self.wavelength:
            self.k = (2 * np.pi) / wavelength_value  # Update k for each wavelength
            power = np.abs(self.transmission_coefficient())**2  # Numerical power calculation
            powers.append(power)
        
        plt.plot(self.wavelength, powers)  # Plot power vs. wavelength
