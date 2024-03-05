import sympy as sym
from IPython.display import display, Math , Latex, Markdown
import numpy as np

def convert_to_latex(x,before="",after="",start_and_end_symbol=""):
    # x can be one expression, a list of expressions, or a two dimensional list of expressions
    b=before.replace(r" ", r"\ ")
    a=after.replace(r" ", r"\ ")
    latex = ''
    justify_symbol=""
    begin_latex_matrix=""
    end_latex_matrix=""
    match np.ndim(x):
        case 0:
            if isinstance(x,sym.Expr):
                latex= sym.latex(x)
            else:
                latex = f'{x}'
        case 1:
            begin_latex_matrix=r'\begin{bmatrix}\begin{array}'
            end_latex_matrix=r'\end{array}\end{bmatrix}'
            for element in x:
                try:
                    if isinstance(element, sym.Expr):
                        latex += sym.latex(element)+"&"
                    else:
                        latex = f'{element}&'
                except TypeError:
                    latex += f'err &'
                # matrix = matrix[:-1] + r'\\'
        case 2:
            justify_symbol=len(x)*"l"
            begin_latex_matrix = r'\begin{bmatrix}\begin{array}'
            end_latex_matrix = r'\end{array}\end{bmatrix}'
            if isinstance(x, sym.Basic):
                for m in range(x.rows):
                    for n in range(x.cols):
                        try:
                            element =x[m,n]
                            if isinstance(element, sym.Expr):
                                latex += sym.latex(element) + "&"
                            else:
                                latex += f'{element}&'
                        except TypeError:
                            latex += f'error &'
                    latex = latex[:-1] + r'\\'

            else:
                for row in x:
                    try:
                        for element in row:
                            if isinstance(element, sym.Expr):
                                latex += sym.latex(element)+"&"
                            else:
                                latex += f'{element}&'
                    except TypeError:
                        latex += f'error &'
                    latex = latex[:-1] + r'\\'
        case _:
            print("none")
    justify_symbol=r"{"+justify_symbol+r"}"
    latex=start_and_end_symbol+ b+begin_latex_matrix+justify_symbol+latex+end_latex_matrix+a+start_and_end_symbol

    return latex
class TaylorSeriesAndOptimization:
    # this class calculates taylor series, gradient and Hessian  of f
    # f is a string defining the function


    def __init__(self,f):
        self.f_symbolic = sym.sympify(f)
        # Convert the set to a sorted list
        self.list_of_individual_symbolic_variables = sorted(self.f_symbolic.free_symbols, key=str)
        self.symbolic_variables=sym.Matrix(self.list_of_individual_symbolic_variables)
        self.list_of_individual_g_symbolic = []
        self.list_of_individual_h_symbolic = []
        for k, symbol in enumerate(self.list_of_individual_symbolic_variables):
            temp_g = sym.diff(self.f_symbolic, symbol)
            self.list_of_individual_g_symbolic.append([temp_g])
            h_row_symbol = []
            for symbol2 in self.list_of_individual_symbolic_variables:
                temp_h = sym.diff(temp_g, symbol2)
                h_row_symbol.append(temp_h)
            self.list_of_individual_h_symbolic.append(h_row_symbol)
        self.g_symbolic=sym.sympify(sym.Matrix(self.list_of_individual_g_symbolic))
        self.h_symbolic=sym.sympify(sym.Matrix(self.list_of_individual_h_symbolic))
    def calculate_numerical_values(self, current_x):
        # x: an N by 1 numpy array representing the initial point
        self.current_x_values=current_x
        self.symbol_value_dict = {}
        for k, symbol in enumerate(self.list_of_individual_symbolic_variables):
            self.symbol_value_dict[symbol] = self.current_x_values[k][0]

        self.symbolic_x_values=sym.Matrix(self.current_x_values)
        self.f_value = sym.sympify(self.f_symbolic.evalf(subs=self.symbol_value_dict))
        self.g_value_sym= sym.sympify(self.g_symbolic.evalf(subs=self.symbol_value_dict))
        self.g_value= np.array(self.g_value_sym.tolist()).astype(np.float64)
        self.h_value_sym= sym.sympify(self.h_symbolic.evalf(subs=self.symbol_value_dict))
        self.h_value = np.array(self.h_value_sym.tolist()).astype(np.float64)
        self.symbolic_x_minus_x_values_sym=sym.sympify(self.symbolic_variables-self.symbolic_x_values)
        temp=sym.sympify(self.h_value_sym * (self.symbolic_x_minus_x_values_sym))
        self.taylor_second_order_sym = self.f_value + self.g_value_sym.transpose().dot(
            self.symbolic_x_minus_x_values_sym) + sym.sympify(self.symbolic_x_minus_x_values_sym.transpose().dot(temp))
    def calculate_alpha_minimize_along_a_line(self,direction):
        # direction: an N by 1 numpy array
        self.alpha=-((self.g_value.transpose().dot(direction))/(direction.transpose().dot(self.h_value.dot(direction))))
        a=1
    def calculate_next_step(self,alpha,direction):
        # direction: an N by 1 numpy array
        self.next_x_values= self.current_x_values + alpha * direction
    def display_f_gradient_and_hessian(self):
        display(Markdown("Function Value: " + convert_to_latex(self.f_value, r"f = ", "", "$")))
        display(Markdown("Gradient: " + convert_to_latex(self.g_symbolic, r"g = ", "", "$")))
        display(Markdown("Gradient at $x_0$ : " + convert_to_latex(self.g_value, r"g = ", "", "$")))
        display(Markdown("Hessian : " + convert_to_latex(self.h_symbolic, r"H = ", "", "$")))
        display(Markdown("Hessian at $X_0$ : " + convert_to_latex(self.h_value, r"H = ", "", "$")))
    def display_second_order_taylor_series(self):
        display(Markdown("Second order Taylor series around $X_0$: " + convert_to_latex(self.taylor_second_order_sym,
                                                                                        r"f(X_0) ~= ", "", "$")))
        display(Markdown("$f\\approx =" + f"{self.f_value}+" + convert_to_latex(self.g_value.transpose()) + \
                         convert_to_latex(self.symbolic_x_minus_x_values_sym) + "+" + \
                         convert_to_latex(self.symbolic_x_minus_x_values_sym.transpose()) + \
                         convert_to_latex(self.h_value) + \
                         convert_to_latex(self.symbolic_x_minus_x_values_sym) + "$"))
    def perform_n_step_gradient_descent_minimize_along_a_line(self,N):
        display(Markdown(r"$\large{\alpha = -\frac{g^{T}P}{P^{T}HP}}$ "))
        display(Markdown(
            r"Direction of P is opposite of gradient. This means: <br/><br/>   $\large{\alpha = -\frac{g^{T}(-g)}{(-g)^{T}H(-g)}}$ "))
        current_x = x0
        for k in range(N):
            self.calculate_numerical_values(current_x)
            self.calculate_alpha_minimize_along_a_line(-self.g_value)
            self.calculate_next_step(self.alpha, -self.g_value)
            display(Markdown(f"Function value at $X_{k}$ : " + convert_to_latex(self.f_value, f"f_{k} = ", "", "$")))
            display(Markdown(f"Gradient at $X_{k}$ : " + convert_to_latex(self.g_value, f"g_{k} = ", "", "$")))
            display(Markdown(f"Hessian at $X_{k}$ : " + convert_to_latex(self.h_value, f"H_{k} = ", "", "$")))
            display(Markdown(convert_to_latex(self.alpha, f" \\alpha_{k} =", "", "$")))
            display(Markdown(f" $X_{k + 1} = X_{k}+\\alpha_{k}*(-g_{k})$"))
            display(Markdown(convert_to_latex(self.next_x_values, f" X_{k + 1} =", r' ', "$")))
            current_x = self.next_x_values

    def perform_n_step_gradient_descent_given_alpha(self, alpha,N):
        current_x = x0
        for k in range(3):
            self.calculate_numerical_values(current_x)
            self.calculate_next_step(alpha, -self.g_value)
            display(Markdown(f"Function value at $X_{k}$ : " + convert_to_latex(self.f_value, f"f_{k} = ", "", "$")))
            display(Markdown(f"Gradient at $X_{k}$ : " + convert_to_latex(self.g_value, f"g_{k} = ", "", "$")))
            display(Markdown(f"Hessian at $X_{k}$ : " + convert_to_latex(self.h_value, f"H_{k} = ", "", "$")))
            display(Markdown(convert_to_latex(alpha, f" \\alpha_{k} =", "", "$")))
            display(Markdown(f" $X_{k + 1} = X_{k}+\\alpha_{k}*(-g_{k})$"))
            display(Markdown(convert_to_latex(self.next_x_values, f" X_{k + 1} =", r' ', "$")))
            current_x = self.next_x_values

    def perform_n_step_minimize_along_a_line_given_direction(self, p,N):
        display(Markdown(r"$\large{\alpha = -\frac{g^{T}P}{P^{T}HP}}$ "))
        current_x = x0
        for k in range(4):
            self.calculate_numerical_values(current_x)
            self.calculate_alpha_minimize_along_a_line(p)
            self.calculate_next_step(self.alpha, p)
            display(Markdown(f"Function value at $X_{k}$ : " + convert_to_latex(self.f_value, f"f_{k} = ", "", "$")))
            display(Markdown(f"Gradient at $X_{k}$ : " + convert_to_latex(self.g_value, f"g_{k} = ", "", "$")))
            display(Markdown(f"Hessian at $X_{k}$ : " + convert_to_latex(self.h_value, f"H_{k} = ", "", "$")))
            display(Markdown(convert_to_latex(self.alpha, f" \\alpha_{k} =", "", "$")))
            display(Markdown(f" $X_{k + 1} = X_{k}+\\alpha_{k}*(-g_{k})$"))
            display(Markdown(convert_to_latex(self.next_x_values, f" X_{k + 1} =", r' ', "$")))
            current_x = self.next_x_values
def solve_assignment_03(f,x0,p,alpha):
    surface=TaylorSeriesAndOptimization(f)
    surface.calculate_numerical_values(x0)
    display(Markdown(r"# <center> Assignment_03 Solution"))
    display(Markdown(r"This is a programmatic solution for Assignment_03.<br> Note that it’s a dynamic solution—you can modify the function equation, starting point, and the number of steps. Simply run this program to compute the new values."))
    display(Markdown(r"Farhad Kamangar  2024<br><br>"))
    display(Markdown(r"# Assignment_03"))
    display(Markdown(r"Consider the following performance surface: <br>"))
    display(Math(r"f(X) = "+ convert_to_latex(surface.f_symbolic)))
    display(Markdown("Assuming initial point:   "+convert_to_latex(x0,r"X_0 = ","","$")))
    display(Markdown("and a given direction:   "+convert_to_latex(p,r"P = ","","$")))
    #
    ##############################################################################################################
    #
    display(Markdown("## • Find the value of the function $f(X)$ , gradient $g$, and Hessian $H$ at $X_0$ <br>"))
    surface.display_f_gradient_and_hessian()
    #
    ##############################################################################################################
    #
    display(Markdown("## • 	Find the Taylor series of this function up to the second derivatives around $X_0 $ <br>"))
    display(Markdown("## • 	Write the Taylor series expansion of the previous step as quadratic function in matrix form.<br>"))
    surface.display_second_order_taylor_series()
    #
    ##############################################################################################################
    #
    display(Markdown("## • 	Perform two steps of gradient descent using minimization along a line (calculate α ) and show the $X_1$ and  $X_2$ positions and the value of the of the function after each step. <br>"))
    surface.perform_n_step_gradient_descent_minimize_along_a_line(3)
    #
    ##############################################################################################################
    #
    display(Markdown(r"## • Perform two steps of gradient descent using $\alpha=0.1$ and show the $X_1$ and  $X_2$ positions and the value of the of the function after each step.. <br>"))
    surface.perform_n_step_gradient_descent_given_alpha(alpha, 3)
    #
    ##############################################################################################################
    #
    display(Markdown(r"## • Perform one step of minimization in the direction of P and show the value $X_1$ and the value of the function after the step.. <br>"))
    surface.perform_n_step_minimize_along_a_line_given_direction(p, 3)
f="2*x1**4-4*x2**4+6*(x1**3)*(x2**2)*(x3**3)-8*x1*x3+10*x3"
x0=np.array([-1,2,3]).reshape(3,1)
p=np.array([4,-5,6]).reshape(3,1)
alpha=0.1
solve_assignment_03(f,x0,p,alpha)