import numpy as np 
import sympy as sp
from sympy import Symbol, symbols, Matrix
from sympy.utilities.autowrap import ufuncify
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

t, r, L, delta, k, omega = symbols("t r L delta k omega")

generic_mirror_matrix = (1 / (1j * t)) * Matrix([[-1, -r], [r, 1]])
generic_propagation_matrix0 = Matrix([[sp.exp(1j * k * (L + delta)), 0], [0, sp.exp(-1j * k * (L + delta))]])
generic_propagation_matrix = generic_propagation_matrix0.subs([(k, omega / 3e8)])

class Mirror:
    """Pass the big T, symbolic in sp sense"""
    def __init__(self, t):
        self.t = sp.sqrt(t)
        self.r = sp.sqrt(1 - self.t ** 2)
    
    def get_matrix(self):
        return generic_mirror_matrix.subs([(r, self.r), (t, self.t)])

    def __repr__(self): 
        return f"Transmissivity = {self.t}, Reflectivity = {self.r}"

class Subcavity:
    """meant to use symbolically; subs_dict should take a form of L1, L2 etc"""
    def __init__(self, subs_dict): 

        self.subs_dict = subs_dict
        self.L = self.subs_dict.get(L, L)
        self.delta = self.subs_dict.get(delta, delta)
        self.k = self.subs_dict.get(k, k) if k is not None else None
        self.total_L = self.L + self.delta

    def get_matrix(self):
        new_subs = {L: self.L, delta: self.delta, k: self.k}
        subbed_expression = generic_propagation_matrix.subs(new_subs)
        return subbed_expression

    def __repr__(self): 
        return f"L = {self.L}, delta = {self.delta}"

class Cavity_sym:
    def __init__(self, mirror_transmissivities, subs_dict): 
        if len(mirror_transmissivities) != len(subs_dict) + 1:
            raise ValueError("Number of mirrors should be one more than the number of subcavities.")

        # ennumerates subcavities starting with 1 !!!!
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

    def transmitted_power(self):
        """Outputs the expression or function of transmitted power with substitutions and lambdification."""
        transfer_matrix = self.transfer_matrix()
        transmitted_power_expression = sp.Abs((transfer_matrix[0, 0]*transfer_matrix[1, 1] - transfer_matrix[0, 1]*transfer_matrix[1, 0]) / transfer_matrix[1, 1])**2  
        return transmitted_power_expression

    def phase_response(self):
        transfer_matrix = self.transfer_matrix()
        transmission_coeff = (transfer_matrix[0, 0]*transfer_matrix[1, 1] - transfer_matrix[0, 1]*transfer_matrix[1, 0]) / transfer_matrix[1, 1]
        phase = sp.arg(transmission_coeff)
        return phase

    def phase_response_reflected(self):
        transfer_matrix = self.transfer_matrix()
        reflection_coeff = -transfer_matrix[1, 0] / transfer_matrix[1, 1]
        phase_ref = sp.arg(reflection_coeff)
        return phase_ref


class Cavity_num:
    """numerical processing, subs_dict ought to contain actual numerical values"""
    def __init__(self, mirror_tr, mirror_vals, subs_dict_symbolic, subs_dict_numerical, lambdify_vars):
        self.subs_dict_symbolic = subs_dict_symbolic
        self.subs_dict_numerical = subs_dict_numerical
        self.mirror_tr = mirror_tr
        self.mirror_vals = mirror_vals
        self.lambdify_vars = lambdify_vars
        self.cavity_sym = Cavity_sym(self.mirror_tr, self.subs_dict_symbolic)

    def dynamic_range(self, values_range):
        """returns a new range in angular frequency"""
        L_values = {key: value for key, value in self.subs_dict_numerical.items() if key.name.startswith("L")}
        delta_values = {key: value for key, value in self.subs_dict_numerical.items() if key.name.startswith("delta")}
        fsr_1 = 3e8 / (2 * (np.sum(list(L_values.values())) + np.sum(list(delta_values.values()))))
        fsr_0 = 3e8 / (2 * (np.sum(list(L_values.values()))))
        ref_nu = 3e8/(1064e-9)
        m = np.floor(ref_nu / fsr_0).astype(int)
        delta_f = m * (fsr_0 - fsr_1)
        nu_sweep = values_range - delta_f
        return (nu_sweep * 2 * np.pi), (delta_f * 2 * np.pi)

    def resonant_frequencies(self, transmission, frequencies):

        peaks, _ = find_peaks(transmission)

        # Get corresponding frequencies
        peak_freq = frequencies[peaks]

        # Sort peaks by proximity to zero
        sorted_peaks = sorted(peak_freq, key=lambda x: abs(x))

        # Select the two closest to zero
        peak_1, peak_2 = sorted_peaks[:2]

        return peak_1, peak_2


    def transmitted_power(self, ranges):
        """takes a range in herz and outputs in angular freq"""
        transmitted_power_expr = self.cavity_sym.transmitted_power()
        transmitted_power_subbed = transmitted_power_expr.subs(self.subs_dict_numerical)
        transmitted_power_subbed1 = transmitted_power_subbed.subs(self.mirror_vals)
        # transmitted_power_subbed2 = transmitted_power_subbed1.subs([(k, omega / 3e8)])
        transmitted_power_function = sp.lambdify(self.lambdify_vars, transmitted_power_subbed1)
        omega_values = np.array(ranges[0])  # Ensure it's an array
        remaining_params = [np.array(p) for p in ranges[1:]]

        omega_values_shifted, shift = self.dynamic_range(omega_values)


        if len(remaining_params) > 0:  
            values = transmitted_power_function(omega_values_shifted, *remaining_params)
        else:
            values = transmitted_power_function(omega_values_shifted)

        resonance_1, resonance_2 = self.resonant_frequencies(values, omega_values_shifted - shift)

        return omega_values_shifted - shift, values, resonance_1, resonance_2
    
    def phase_response(self, ranges):
        phase_response_expr = self.cavity_sym.phase_response()
        phase_response_subbed = phase_response_expr.subs(self.subs_dict_numerical)
        phase_response_subbed1 = phase_response_subbed.subs(self.mirror_vals)
        phase_response_function = sp.lambdify(self.lambdify_vars, phase_response_subbed1)

        omega_values = np.array(ranges[0])  # Ensure it's an array
        remaining_params = [np.array(p) for p in ranges[1:]]

        omega_values_shifted, shift = self.dynamic_range(omega_values)

        if len(remaining_params) == 0:  
            values = phase_response_function(omega_values_shifted)
        else:
            values = phase_response_function(omega_values_shifted, *remaining_params)

        return (omega_values_shifted - shift, values, phase_response_function)

    def folded_spectrum(self, ranges, detuning):
        """k has to go first in lamdbify vars"""
        phase_response_expr = self.cavity_sym.phase_response()
        phase_response_subbed = phase_response_expr.subs(self.subs_dict_numerical)
        phase_response_subbed1 = phase_response_subbed.subs(self.mirror_vals)
        omega_values = np.array(ranges[0]) * 2 * np.pi  # Ensure it's an array
        remaining_params = [np.array(p) for p in ranges[1:]]

        phase_response_function = sp.lambdify(self.lambdify_vars, phase_response_subbed1, 'numpy')


        omega_values = np.array(ranges[0])  # Ensure it's an array
        remaining_params = [np.array(p) for p in ranges[1:]]

        omega_values_shifted, _, res_1, res_2 = self.transmitted_power([omega_values])

        # fix unit of detuning
        omega_plus_values = omega_values_shifted - (detuning * 2 * np.pi) - res_1
        omega_minus_values = -omega_values_shifted - (detuning * 2 * np.pi) - res_1

        if len(remaining_params) == 0:  
            folded_spectrum = (phase_response_function(omega_plus_values) + phase_response_function(omega_minus_values)) / 2
        else:
            folded_spectrum = (phase_response_function(omega_plus_values, *remaining_params) + phase_response_function(omega_minus_values, *remaining_params)) / 2

        return omega_values_shifted, folded_spectrum


    def phase_response_reflected(self, ranges):
            phase_response_expr = self.cavity_sym.phase_response_reflected()
            phase_response_subbed = phase_response_expr.subs(self.subs_dict_numerical)
            phase_response_subbed1 = phase_response_subbed.subs(self.mirror_vals)
            phase_response_function = sp.lambdify(self.lambdify_vars, phase_response_subbed1)

            omega_values = np.array(ranges[0])  # Ensure it's an array
            remaining_params = [np.array(p) for p in ranges[1:]]

            omega_values_shifted, shift = self.dynamic_range(omega_values)

            if len(remaining_params) == 0:  
                values = phase_response_function(omega_values_shifted)
            else:
                values = phase_response_function(omega_values_shifted, *remaining_params)

            return (omega_values_shifted - shift, values, phase_response_function)

    def folded_spectrum_reflected(self, ranges, detuning):
            """k has to go first in lamdbify vars"""
            phase_response_expr = self.cavity_sym.phase_response_reflected()
            phase_response_subbed = phase_response_expr.subs(self.subs_dict_numerical)
            phase_response_subbed1 = phase_response_subbed.subs(self.mirror_vals)
            omega_values = np.array(ranges[0]) * 2 * np.pi  # Ensure it's an array
            remaining_params = [np.array(p) for p in ranges[1:]]

            phase_response_function = sp.lambdify(self.lambdify_vars, phase_response_subbed1, 'numpy')


            omega_values = np.array(ranges[0])  # Ensure it's an array
            remaining_params = [np.array(p) for p in ranges[1:]]

            omega_values_shifted, _, res_1, res_2 = self.transmitted_power([omega_values])

            # fix unit of detuning
            omega_plus_values = omega_values_shifted - (detuning * 2 * np.pi) - (res_1)
            omega_minus_values = -omega_values_shifted - (detuning * 2 * np.pi) - (res_1)

            if len(remaining_params) == 0:  
                folded_spectrum = (phase_response_function(omega_plus_values) + phase_response_function(omega_minus_values)) / 2
            else:
                folded_spectrum = (phase_response_function(omega_plus_values, *remaining_params) + phase_response_function(omega_minus_values, *remaining_params)) / 2

            return omega_values_shifted, folded_spectrum
    
    def folded_spectrum_reflected_derivative(self, ranges, detuning):
            """k has to go first in lamdbify vars"""
            phase_response_expr = self.cavity_sym.phase_response_reflected()
            phase_response_subbed = phase_response_expr.subs(self.subs_dict_numerical)
            phase_response_subbed1 = phase_response_subbed.subs(self.mirror_vals)
            omega_values = np.array(ranges[0]) * 2 * np.pi  # Ensure it's an array
            remaining_params = [np.array(p) for p in ranges[1:]]

            phase_response_function = sp.lambdify(self.lambdify_vars, phase_response_subbed1, 'numpy')


            omega_values = np.array(ranges[0])  # Ensure it's an array
            remaining_params = [np.array(p) for p in ranges[1:]]

            omega_values_shifted, _, res_1, res_2 = self.transmitted_power([omega_values])

            # fix unit of detuning
            omega_plus_values = omega_values_shifted - (detuning * 2 * np.pi) - res_1
            omega_minus_values = -omega_values_shifted - (detuning * 2 * np.pi) - res_1

            if len(remaining_params) == 0:  
                folded_spectrum = (phase_response_function(omega_plus_values) + phase_response_function(omega_minus_values)) / 2
            else:
                folded_spectrum = (phase_response_function(omega_plus_values, *remaining_params) + phase_response_function(omega_minus_values, *remaining_params)) / 2

            derivative_folded_spectrum = np.gradient(folded_spectrum, omega_values_shifted)

            return omega_values_shifted, derivative_folded_spectrum

