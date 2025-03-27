import numpy as np 
import sympy as sp
from sympy import Symbol, symbols, Matrix
import matplotlib.pyplot as plt

t, r, L, delta, k, wavelength = symbols("t r L delta k lambda1")

# try to change this such that it works SOLELY in symbols so that it first computes everything symbollically

generic_mirror_matrix = (1 / (1j * t)) * Matrix([[-1, -r], [r, 1]])
generic_propagation_matrix = Matrix([[sp.exp(1j * k * (L + delta)), 0], [0, sp.exp(-1j * k * (L + delta))]])

class Mirror:
    """Pass the big T, can be either symbolic in sp sense or a number"""
    def __init__(self, t):
        self.t = sp.sqrt(t)
        self.r = sp.sqrt(1 - self.t ** 2)
    
    def get_matrix(self):
        return generic_mirror_matrix.subs([(r, self.r), (t, self.t)])

    def __repr__(self): 
        return f"Transmissivity = {self.t}, Reflectivity = {self.r}"


class Subcavity:
    def __init__(self, subs_dict, lambdify_vars = None): 

        self.subs_dict = subs_dict
        self.lambdify_vars = lambdify_vars if lambdify_vars is not None else None
        self.L = self.subs_dict.get(L, L)
        self.delta = self.subs_dict.get(delta, delta)
        self.wavelength = self.subs_dict.get(wavelength, wavelength)
        self.k = (2 * sp.pi) / self.wavelength  if self.wavelength is not None else None
        self.total_L = self.L + self.delta

    def get_matrix(self):
        new_subs = {L: self.L, delta: self.delta, k: self.k}
        subbed_expression = generic_propagation_matrix.subs(new_subs)

        if self.lambdify_vars is None:
            return subbed_expression
        else:
            func = sp.lambdify(self.lambdify_vars, subbed_expression)
            return func


class Cavity:
    """
    Initializes a cavity with subcavities and mirrors.

    Parameters:
    - subcavity_params: List of (L, delta) tuples for each subcavity.
    - mirror_transmissivities: List of transmissivity values for mirrors separating subcavities.
    - k: Wave number (constant for all subcavities).
    """
    def __init__(self, mirror_transmissivities, subs_dict): 
        if len(mirror_transmissivities) != len(subs_dict) + 1:
            raise ValueError("Number of mirrors should be one more than the number of subcavities.")

        # ennumerates subcavities starting with 1 !!!!
        self.wavelength = wavelength
        self.k = (2 * sp.pi) / wavelength  if wavelength is not None else None
        self.subcavities = {
            f"subcavity_{i+1}": Subcavity(subs_dict=subs_dict[i])
            for i in range(len(subs_dict))
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
        total_matrix = sp.eye(2)
        matrices = [self.get_mirror_matrix(len(self.mirrors))]
        for i in range(len(self.subcavities), 0, -1):  # From N to 1
            matrices.append(self.get_propagation_matrix(i))  # Add propagation matrix
            matrices.append(self.get_mirror_matrix(i))  # Add mirror matrix
        
        for matrix in matrices:
            total_matrix = total_matrix * matrix

        return total_matrix

    def transmitted_power(self, subs_dict, lambdify_vars = None):
        """Outputs the expression or function of transmitted power with substitutions and lambdification."""
        transfer_matrix = self.transfer_matrix()
        transmitted_power_expression = sp.Abs((transfer_matrix[0, 0]*transfer_matrix[1, 1] - transfer_matrix[0, 1]*transfer_matrix[1, 0]) / transfer_matrix[1, 1])**2  
        substituted_expression = transmitted_power_expression.subs(subs_dict)
        
        if lambdify_vars:
            return sp.lambdify(lambdify_vars, substituted_expression, 'numpy')
        return substituted_expression


    def transmitted_power_plot(self, subs_dict, lambdify_vars, val_range):
        """Computation of the transmitted power, outputs the frequency range and values"""
        self.subs_dict = subs_dict
        self.lambdify_vars = lambdify_vars
        self.val_range = val_range

        def dynamic_range(self, value_range):
            L_values = {key: value for key, value in self.subs_dict.items() if key.name.startswith("L")}
            delta_values = {key: value for key, value in self.subs_dict.items() if key.name.startswith("delta")}
            fsr_1 = 3e8 / (2 * (np.sum(list(L_values.values())) + np.sum(list(delta_values.values()))))
            fsr_0 = 3e8 / (2 * (np.sum(list(L_values.values()))))
            ref_nu = 3e8/(1064e-9)
            m = np.floor(ref_nu / fsr_0).astype(int)
            delta_f = m * (fsr_0 - fsr_1)

            # resonance_nu = fsr_0 * m + delta_f
            # nu_bottom = resonance_nu - 75e6
            # nu_top = resonance_nu + 75e6
            nu_sweep = self.val_range - delta_f
            # np.linspace(nu_bottom, nu_top, 10000)
            k_range = nu_sweep * 2 * np.pi /3e8
            return k_range, nu_sweep, delta_f


        func = self.transmitted_power(self.subs_dict, self.lambdify_vars)
        k_range, nu_sweep, delta_f = dynamic_range(self, self.val_range)
        values = func(k_range)
        # plt.plot((nu_sweep - delta_f) * 1e-6, values)
        return (nu_sweep - delta_f), values

    def phase_response(self, subs_dict, lambdify_vars, detuning, val_range):
        self.val_range = val_range
        transfer_matrix = self.transfer_matrix()
        transmission_coeff = (transfer_matrix[0, 0]*transfer_matrix[1, 1] - transfer_matrix[0, 1]*transfer_matrix[1, 0]) / transfer_matrix[1, 1]
        self.subs_dict = subs_dict
        self.lambdify_vars = lambdify_vars
        self.detuning = detuning

        lambda1 = Symbol("lambda1")
        
        if lambda1 in self.lambdify_vars:
            self.transmission_coeff_subbed = transmission_coeff.subs(self.subs_dict)
            transmission_coeff_func = sp.lambdify(self.lambdify_vars, self.transmission_coeff_subbed)
            k_range = self.val_range * 2 * np.pi / 3e8

            phase = np.angle((transmission_coeff_func(k_range - detuning) + transmission_coeff_func(- k_range - detuning)/2))

        # plt.plot(self.val_range, phase)
            return  self.val_range, phase, transmission_coeff_func
        
        else: # detuning in wavelength
            
            self.transmission_coeff_subbed = transmission_coeff.subs(self.subs_dict)
            transmission_coeff_func = sp.lambdify(self.lambdify_vars, self.transmission_coeff_subbed)
            phase = np.angle((transmission_coeff_func(2 * np.pi / (lambda1) - detuning) + transmission_coeff_func(- 2 * np.pi / (k_range) - detuning)/2))
            return self.val_range, phase, transmission_coeff_func


    def phase_response_whittle(self, t1, cavity_detuning, omega_range, subs):
        self.t1 = t1
        self.cavity_detuning = cavity_detuning
        self.omega_range = omega_range

        omega = Symbol("omega")
        mir_1_t = Symbol("t1")

        total_length = self.total_length.subs(subs)

        gamma = 3e8 * mir_1_t / (4 * total_length)

        nominator = 2 * gamma * (omega - self.cavity_detuning)
        denominator = - gamma **2 + (omega - self.cavity_detuning) **2
    
        alpha_fc = sp.atan(nominator / denominator)
        alpha_fc = alpha_fc.subs(mir_1_t, self.t1)

        alpha_fc_function = sp.lambdify((omega), alpha_fc)

        def alpha_p(self, omega, t):
            return (alpha_fc_function(omega) + alpha_fc_function(- omega)) / 2
        
        alpha_p_arr = alpha_p(self, self.omega_range, self.t1)
        values = np.unwrap(alpha_p_arr, period = np.pi/4, axis = 0)
        # plt.plot(self.omega_range, np.unwrap(alpha_p_arr, period = np.pi/4, axis = 0))
        return self.omega_range, values

    def target_phase_response(self, power, mass, finess, L, omega_m, Q, wavelength, omega_arr):
        omega_laser = Symbol("Omega_laser")
        chi = Symbol("Chi")
        omega = Symbol('Omega')
        
        c = 3e8

        self.power = power
        self.mass = mass
        self.finess = finess
        self.omega_m = omega_m
        self.Q = Q
        self.L = L
        self.damping = self.omega_m / self.Q
        self.wavelength = wavelength
        self.omega_arr = omega_arr
        fsrOM = c / 2 * self.L
        self.kappaOM = fsrOM / self.finess


        chi = ( self.mass * (self.omega_m ** 2 - omega ** 2 + 1j * self.damping * omega)) ** (-1)

        omega_laser = 2 * np.pi * c / self.wavelength
        K = sp.atan((16 * omega_laser * self.power) / (c * L * self.kappaOM) * chi)
        K_func = sp.lambdify(omega, K)

        values = K_func(omega_arr)

        # plt.plot(omega_arr, values)

        return omega_arr, values, K_func