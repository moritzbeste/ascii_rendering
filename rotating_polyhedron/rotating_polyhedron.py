import numpy as np
import math
import time
import os
import sys

class Polyhedron:
    def __init__(self, shape='cube', side_length=29, aspect_ratio=1.67, draw_faces=False):
        self.camera_vector = np.array([0, 0, -1])
        self.draw_faces = draw_faces
        self.lookup_symbols = np.array([' ', ':', ';', '!', '<', '~', '-', '+', '?', '/', '|', '*', '$', '%', '#', '@'])
        self.lookup_black = len(self.lookup_symbols) - 1
        self.aspect_ratio = aspect_ratio
        self.shape = shape
        self.side_length = side_length
        self.polyhedron, self.render = self.generate_polyhedron_and_render()
        self.c = np.mean(self.polyhedron, axis=0)
        self.polyhedron_offset = self.polyhedron - self.c
        if draw_faces: 
            self.draw_method = self.render_polyhedron_faces
            self.lookup_faces = {'cube' : 
                np.array([[0, 4, 6, 2], [2, 6, 7, 3], [4, 5, 7, 6], [1, 3, 7, 5], [0, 1, 5, 4], [0, 2, 3, 1]]),
                'octahedron' : 
                np.array([[0, 2, 4], [2, 5, 3], [2, 3, 4], [0, 5, 2], [1, 3, 5], [0, 4, 1], [0, 1, 5], [1, 4, 3]])}
            if self.shape not in self.lookup_faces:
                raise NotImplementedError(f"Dynamic lighting for shape {self.shape} not supported.")
            self.faces = self.lookup_faces[self.shape]
        else:
            self.draw_method = self.render_polyhedron_edges
            self.lookup_edges = {'cube' : 
                np.array([[0, 1], [0, 2], [0, 4],
                [3, 1], [3, 2], [3, 7],
                [5, 1], [5, 4], [5, 7],
                [6, 2], [6, 4], [6, 7]]),
                'octahedron' :
                np.array([[0, 1], [0, 2], [3, 1], [3, 2],
                [4, 0],  [4, 1], [4, 2], [4, 3],
                [5, 0],  [5, 1], [5, 2], [5, 3]])}
            if self.shape not in self.lookup_edges:
                raise NotImplementedError(f"edges for shape {self.shape} not supported.")
            self.edges = self.lookup_edges[self.shape]


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


    def generate_cube_corners(self, higher, lower):
        return np.array([
                [higher, higher, higher],
                [higher, higher, lower],
                [higher, lower, higher],
                [higher, lower, lower],
                [lower, higher, higher],
                [lower, higher, lower],
                [lower, lower, higher],
                [lower, lower, lower]])


    def generate_octahedron_corners(self, higher, lower):
        return np.array([
                [(higher + lower) // 2, higher, higher],
                [(higher + lower) // 2, higher, lower],
                [(higher + lower) // 2, lower, higher],
                [(higher + lower) // 2, lower, lower],
                [higher, (higher + lower) // 2, (higher + lower) // 2],
                [lower, (higher + lower) // 2, (higher + lower) // 2]])


    def generate_polyhedron_and_render(self):
        if self.side_length % 2 != 1:
            self.side_length += 1
        render_dim = (math.ceil(self.side_length * np.sqrt(3)), math.ceil(self.side_length * np.sqrt(3)))
        dist_to_center = self.side_length // 2
        lower = render_dim[0] // 2 - dist_to_center
        higher = render_dim[0] // 2 + dist_to_center
        
        if self.shape == 'cube':
            polyhedron = self.generate_cube_corners(higher, lower)
        elif self.shape == 'octahedron':
            polyhedron = self.generate_octahedron_corners(higher, lower)

        render = np.zeros((render_dim[0], math.ceil(render_dim[1] * self.aspect_ratio)), dtype=int)
        return polyhedron, render


    def clear_terminal(self):
        os.system('cls' if os.name == 'nt' else 'clear')


    def render_polyhedron_edges(self, polyhedron):
        polyhedron[:, 0] *= self.aspect_ratio
        for endpoint_0, endpoint_1 in self.edges:
            dx = np.abs(polyhedron[endpoint_1][0] - polyhedron[endpoint_0][0])
            dy = np.abs(polyhedron[endpoint_1][1] - polyhedron[endpoint_0][1])
            angle = 2 * np.abs(np.arctan2(dy, dx)) / np.pi
            angle = 1 - angle
            density_modifier = 1 + (self.aspect_ratio - 1) * angle
            
            density = np.linspace(0, 1, num=int(self.side_length * density_modifier)).reshape(-1, 1)
            points = np.round(polyhedron[endpoint_0] + density * (polyhedron[endpoint_1] - polyhedron[endpoint_0])).astype(int)
            self.render[np.clip(points[:, 1], 0, self.render.shape[0] - 1), np.clip(points[:, 0], 0, self.render.shape[1] - 1)] = self.lookup_black
    

    def triangulate_convex_polygon(self, face):
        # fan triangulation of convex polygon
        triangles = []
        n = len(face)
        if n < 3:
            raise ValueError("Polygon has less than 3 verticies.")
        for i in range(1, n - 1):
            triangles.append((face[0], face[i], face[i + 1]))
        return triangles


    def fill_triangle(self, polyhedron, triangle, shade_index):
        # bounding box for triangle
        p0 = polyhedron[triangle[0]]
        p1 = polyhedron[triangle[1]]
        p2 = polyhedron[triangle[2]]
        min_x = int(max(min(p0[0], p1[0], p2[0]), 0))
        max_x = int(min(max(p0[0], p1[0], p2[0]), self.render.shape[1] - 1))
        min_y = int(max(min(p0[1], p1[1], p2[1]), 0))
        max_y = int(min(max(p0[1], p1[1], p2[1]), self.render.shape[0] - 1))

        xs, ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        points = np.stack((xs, ys), axis=-1)
        
        def edge(a, b, p):
            return (p[..., 0] - a[0]) * (b[1] - a[1]) - (p[..., 1] - a[1]) * (b[0] - a[0])

        w0 = edge(p1, p2, points)
        w1 = edge(p2, p0, points)
        w2 = edge(p0, p1, points)

        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        ys, xs = np.where(inside)
        self.render[min_y + ys, min_x + xs] = shade_index


    def render_polyhedron_faces(self, polyhedron):
        polyhedron[:, 0] *= self.aspect_ratio
        for face in self.faces:
            # we only need 3 points to calculate the normal vector and they should be indexed in counter clockwise orientation
            p1 = polyhedron[face[0]]
            p2 = polyhedron[face[1]]
            p3 = polyhedron[face[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            normal_vector = np.cross(v1, v2)
            normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)

            # check if the face is facing away
            dot_product = np.dot(normalized_normal_vector, self.camera_vector)
            if dot_product < 0:
                triangles = self.triangulate_convex_polygon(face)
                symbol_index = np.clip(a=int(len(self.lookup_symbols) * dot_product * -1), a_min=0, a_max=len(self.lookup_symbols) - 1)
                for triangle in triangles:
                    self.fill_triangle(polyhedron, triangle, shade_index=symbol_index)
                    

    def print_render(self, polyhedron):
        self.draw_method(polyhedron)
        char_matrix = self.lookup_symbols[self.render]
        self.render.fill(0)
        self.clear_terminal()
        for row in char_matrix:
            print(''.join(row))


    def consistently_rotate_polyhedron(self, theta=np.array([0.1, 0.01, 0.05])):
        total_theta = np.array([0.0, 0.0, 0.0])
        while(True):
            total_theta = (total_theta + theta) % (2 * np.pi)
            rotation_matrix = self.multi_dim_rotation(total_theta)
            temp_polyhedron = self.polyhedron_offset @ rotation_matrix + self.c
            self.print_render(polyhedron=temp_polyhedron)
            time.sleep(0.05)


if __name__ == '__main__':
    n = len(sys.argv)
    if n != 7:
        side_length = 29 
        theta = np.array([0.1, 0.01, 0.05])
        shape = 'cube'
        draw_faces = 1
    else:
        side_length = int(sys.argv[1])
        theta = np.array([float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])])
        shape = str(sys.argv[5])
        draw_faces = str(sys.argv[6])
    poly = Polyhedron(shape=shape, side_length=29, draw_faces=draw_faces)
    poly.consistently_rotate_polyhedron(theta)
