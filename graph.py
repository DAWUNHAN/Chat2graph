
from manim import *
import numpy as np
from sympy import symbols, solve, log

class Graph(Scene):
    def construct(self):
        # Define the axes
        axes = Axes(
            x_range=[0.1, 10, 1],  # Avoid zero to prevent log issues
            y_range=[-5, 5, 1],
            axis_config={"color": BLUE}
        )

        # Define the functions
        def f(x):
            return np.log(x / 2)

        def g(x):
            return (x - 8) / 2

        # Define the x values for plotting
        x_vals = np.linspace(0.1, 10, 1000)  # Avoid zero to prevent log issues

        # Plot the functions
        f_graph = axes.plot(f, x_range=[0.1, 10], color=YELLOW)
        g_graph = axes.plot(g, x_range=[0.1, 10], color=GREEN)

        # Add axes and graphs to the scene
        self.play(Create(axes))
        self.play(Create(f_graph), Create(g_graph))

        # Find intersections using sympy
        x = symbols('x')
        intersections = solve(log(x / 2) - (x - 8) / 2, x)
        intersections = [float(i) for i in intersections if i > 0]  # Filter valid intersections

        # Plot intersection points
        intersection_dots = VGroup()
        for x_val in intersections:
            y_val = f(x_val)
            dot = Dot(axes.coords_to_point(x_val, y_val), color=RED)
            intersection_dots.add(dot)
            self.play(Create(dot))

        # Shade the area between the intersections
        if len(intersections) == 2:
            x_min, x_max = intersections
            shaded_area = axes.get_area(f_graph, x_range=(x_min, x_max), bounded_graph=g_graph, color=BLUE, opacity=0.5)
            self.play(Create(shaded_area))

        self.wait(2)

# To run the scene, use the following command in the terminal:
# manim -pql script_name.py Graph
