import numpy as np 
import sympy as sp
import matplotlib.pyplot as plt


x, y = sp.symbols("x y")

expression = x **2 + y**3

class Test:
    def __init__(self, subs_dict, lambdify_vars):
        
        self.subs_dict = subs_dict
        self.lambdify_vars = lambdify_vars
    
    def process_expression(self):
        substituted = expression.subs(self.subs_dict)
        func = sp.lambdify(self.lambdify_vars, substituted)
        return func

    def plot(self, x):
        function = self.process_expression()
        values = function(x)
        plt.plot(x, values)


