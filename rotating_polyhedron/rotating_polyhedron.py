import numpy as np
import math
import time
import os
import sys

class Polyhedron:
    def __init__(self, shape="cube", side_length=29, aspect_ratio = 1.67):
        self.aspect_ratio = aspect_ratio
        self.shape = shape
        self.side_length = side_length
        self.polyhedron, self.screen = self.generate_polyhedron_and_screen()
        self.c = np.mean(self.polyhedron, axis=0)
        self.polyhedron_offset = self.polyhedron - self.c
        lookup = {"cube" : 
            [(0, 1), (0, 2), (0, 4),
            (3, 1), (3, 2), (3, 7),
            (5, 1), (5, 4), (5, 7),
            (6, 2), (6, 4), (6, 7),],
            "tetrahedron" :
            [(0 , 1), (0, 2),
            (3, 1), (3, 2),
            (4, 0),  (4, 1), (4, 2), (4, 3),
            (5, 0),  (5, 1), (5, 2), (5, 3)]}
        if self.shape not in lookup:
            raise NotImplementedError(f"Shape {self.shape} not supported.")
        self.edges = lookup[self.shape]


    def x_rotation(self, theta_x):
        cos = np.cos(theta_x)
        sin = np.sin(theta_x)
        x_rotation_matrix = np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        return x_rotation_matrix


    def y_rotation(self, theta_y):
        cos = np.cos(theta_y)
        sin = np.sin(theta_y)
        y_rotation_matrix = np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        return y_rotation_matrix


    def z_rotation(self, theta_z):
        cos = np.cos(theta_z)
        sin = np.sin(theta_z)
        z_rotation_matrix = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        return z_rotation_matrix


    def multi_dim_rotation(self, theta_xyz):
        rotation_matrix = self.x_rotation(theta_xyz[0])
        if theta_xyz[1] != 0:
            rotation_matrix = rotation_matrix @ self.y_rotation(theta_xyz[1])
        if theta_xyz[2] != 0:
            rotation_matrix = rotation_matrix @ self.z_rotation(theta_xyz[2])
        return rotation_matrix


    def generate_polyhedron_and_screen(self):
        screen_dim = (math.ceil(self.side_length * np.sqrt(3)), math.ceil(self.side_length * np.sqrt(3)))
        if self.side_length % 2 != 1:
            self.side_length += 1
        dist_to_center = self.side_length // 2
        lower = screen_dim[0] // 2 - dist_to_center
        higher = screen_dim[0] // 2 + dist_to_center
        
        if self.shape == "cube":
            polyhedron = np.array([
                [higher, higher, higher],
                [higher, higher, lower],
                [higher, lower, higher],
                [higher, lower, lower],
                [lower, higher, higher],
                [lower, higher, lower],
                [lower, lower, higher],
                [lower, lower, lower]])
        elif self.shape == "tetrahedron":
            polyhedron = np.array([
                [(higher + lower) // 2, higher, higher],
                [(higher + lower) // 2, higher, lower],
                [(higher + lower) // 2, lower, higher],
                [(higher + lower) // 2, lower, lower],
                [higher, (higher + lower) // 2, (higher + lower) // 2],
                [lower, (higher + lower) // 2, (higher + lower) // 2]])

        screen = np.zeros((screen_dim[0], math.ceil(screen_dim[1] * self.aspect_ratio)), dtype=int)
        return polyhedron, screen


    def clear_terminal(self):
        os.system('cls' if os.name == 'nt' else 'clear')


    def draw_polyhedron(self, polyhedron):
        polyhedron[:, 1] *= self.aspect_ratio
        for endpoint_0, endpoint_1 in self.edges:
            xy = polyhedron[endpoint_0] - polyhedron[endpoint_1]
            if xy[0] == 0:
                y_start = min(polyhedron[endpoint_0][1], polyhedron[endpoint_1][1])
                y_end = max(polyhedron[endpoint_0][1], polyhedron[endpoint_1][1])
                y_values = np.linspace(y_start, y_end, self.side_length)
                self.screen[int(polyhedron[endpoint_0][0]), np.clip(y_values.astype(int), 0, self.screen.shape[1] - 1)] = 1
            else:
                m = xy[1] / xy[0]
                b = polyhedron[endpoint_0][1] - m * polyhedron[endpoint_0][0]
                x_start = min(polyhedron[endpoint_0][0], polyhedron[endpoint_1][0])
                x_end = max(polyhedron[endpoint_0][0], polyhedron[endpoint_1][0])
                x_values = np.linspace(x_start, x_end, int(self.side_length * self.aspect_ratio))
                y_values = m * x_values + b
                self.screen[np.clip(x_values.astype(int), 0, self.screen.shape[0] - 1), np.clip(y_values.astype(int), 0, self.screen.shape[1] - 1)] = 1
                    

    def print_screen(self, polyhedron):
        self.draw_polyhedron(polyhedron)
        lookup = np.array([" ", "@"])
        char_matrix = lookup[self.screen]
        self.screen.fill(0)
        self.clear_terminal()
        for row in char_matrix:
            print(''.join(row))


    def consistently_rotate_polyhedron(self, theta=np.array([0.1, 0.01, 0.05])):
        total_theta = np.array([0.0, 0.0, 0.0])
        while(True):
            total_theta = (total_theta + theta) % (2 * np.pi)
            rotation_matrix = self.multi_dim_rotation(total_theta)
            temp_polyhedron = self.polyhedron_offset @ rotation_matrix + self.c
            self.print_screen(temp_polyhedron)
            time.sleep(0.05)


if __name__ == "__main__":
    n = len(sys.argv)
    if n != 6:
        side_length = 29 
        theta = np.array([0.1, 0.01, 0.05])
        shape = "cube"
    else:
        side_length = int(sys.argv[1])
        theta = np.array([float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])])
        shape = str(sys.argv[5])
    poly = Polyhedron(shape="cube", side_length=29)
    poly.consistently_rotate_polyhedron(theta)
