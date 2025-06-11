import numpy as np
import math
import time
import os
import sys

class Polyhedron:
    def __init__(self, shape='cube', side_length=29, aspect_ratio=1.67, draw_faces=False):
        self.camera_vector = np.array([0, 0, -1])

        self.lookup_symbols = np.array([' ', ':', ';', '!', '-', '~', '+', '<', '?', '/', '|', '*', 'O', '$', '%', '#', '@'])
        self.lookup_black = len(self.lookup_symbols) - 1

        self.aspect_ratio = aspect_ratio
        self.aspect_ratio_transformation_matrix = np.array([[self.aspect_ratio, 0, 0], [0, 1, 0], [0, 0, 1]])

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

            self.triangles = {}
            for face_index, face in enumerate(self.faces):
                self.triangles[face_index] = self.triangulate_convex_polygon(face)
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

            self.density_lookup = {}
            self.num_sections = 64
            for i in range(self.num_sections):
                angle = i / (self.num_sections - 1)
                density_modifier = 1 + (self.aspect_ratio - 1) * angle
                density = np.linspace(0, 1, num=int(self.side_length * density_modifier)).reshape(-1, 1)
                self.density_lookup[i] = density



    # fan triangulation of a convex polygon
    def triangulate_convex_polygon(self, face):
        # triangulation of convex polygon
        # we choose a pivot and then use fan triangulation
        triangles = []
        n = len(face)
        if n < 3:
            raise ValueError("Polygon has less than 3 verticies.")
        for i in range(1, n - 1):
            triangles.append((face[0], face[i], face[i + 1]))
        return triangles


    # generates the rotation matrix for the x element of theta
    def x_rotation(self, theta_x):
        cos = np.cos(theta_x)
        sin = np.sin(theta_x)
        x_rotation_matrix = np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        return x_rotation_matrix


    # generates the rotation matrix for the y element of theta
    def y_rotation(self, theta_y):
        cos = np.cos(theta_y)
        sin = np.sin(theta_y)
        y_rotation_matrix = np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        return y_rotation_matrix


    # generates the rotation matrix for the z element of theta
    def z_rotation(self, theta_z):
        cos = np.cos(theta_z)
        sin = np.sin(theta_z)
        z_rotation_matrix = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        return z_rotation_matrix


    # calculates a full rotation matrix for a 3 dimensional rotation
    def multi_dim_rotation(self, theta_xyz):
        rotation_matrix = self.x_rotation(theta_xyz[0])
        if theta_xyz[1] != 0:
            rotation_matrix = rotation_matrix @ self.y_rotation(theta_xyz[1])
        if theta_xyz[2] != 0:
            rotation_matrix = rotation_matrix @ self.z_rotation(theta_xyz[2])
        return rotation_matrix


    # generates corner points for a cube
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


    # generates corner points for an octahedron
    def generate_octahedron_corners(self, higher, lower):
        return np.array([
                [(higher + lower) // 2, higher, higher],
                [(higher + lower) // 2, higher, lower],
                [(higher + lower) // 2, lower, higher],
                [(higher + lower) // 2, lower, lower],
                [higher, (higher + lower) // 2, (higher + lower) // 2],
                [lower, (higher + lower) // 2, (higher + lower) // 2]])


    # generates a general polyhedron and instantiates the render matrix
    def generate_polyhedron_and_render(self):
        render_dim = (math.ceil(self.side_length * np.sqrt(3)), math.ceil(self.side_length * np.sqrt(3)))
        dist_to_center = self.side_length // 2
        lower = render_dim[0] // 2 - dist_to_center
        higher = render_dim[0] // 2 + dist_to_center
        
        # generate the polyhedron
        if self.shape == 'cube':
            polyhedron = self.generate_cube_corners(higher, lower)
        elif self.shape == 'octahedron':
            polyhedron = self.generate_octahedron_corners(higher, lower)

        # instantiate the render matrix
        render = np.zeros((render_dim[0], math.ceil(render_dim[1] * self.aspect_ratio)), dtype=int)

        return polyhedron, render


    # renders the polyhedron as a wire frame
    def render_polyhedron_edges(self, polyhedron):
        # iterate over the edges of the polyhedron
        for endpoint_0, endpoint_1 in self.edges:
            # we calculate a new density modifier for the lines so that vertical lines have less density and horizontal lines have more density because of the aspect ratio of monospace font
            dx = np.abs(polyhedron[endpoint_1][0] - polyhedron[endpoint_0][0])
            dy = np.abs(polyhedron[endpoint_1][1] - polyhedron[endpoint_0][1])
            angle = 2 * np.abs(np.arctan2(dy, dx)) / np.pi
            inverted_angle = 1 - angle
            angle_section_index = np.clip(np.round(inverted_angle * self.num_sections).astype(int), 0, self.num_sections - 1)
            # lookup closest density in lookup table
            density = self.density_lookup[angle_section_index]

            # calculate points on the edge based on the calculated density
            points = np.round(polyhedron[endpoint_0] + density * (polyhedron[endpoint_1] - polyhedron[endpoint_0])).astype(int)
            # draw the indexes of the symbols in the render matrix
            self.render[np.clip(points[:, 1], 0, self.render.shape[0] - 1), np.clip(points[:, 0], 0, self.render.shape[1] - 1)] = self.lookup_black


    # rendering the triangle
    def fill_triangle(self, polyhedron, triangle, shade_index):
        # bounding box for triangle
        p0 = polyhedron[triangle[0]]
        p1 = polyhedron[triangle[1]]
        p2 = polyhedron[triangle[2]]
        min_x = int(max(min(p0[0], p1[0], p2[0]), 0))
        max_x = int(min(max(p0[0], p1[0], p2[0]), self.render.shape[1] - 1))
        min_y = int(max(min(p0[1], p1[1], p2[1]), 0))
        max_y = int(min(max(p0[1], p1[1], p2[1]), self.render.shape[0] - 1))

        # generate a mesh grid
        xs, ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        points = np.stack((xs, ys), axis=-1)
        
        # for calculating which side of the line points are on
        def edge(a, b, p):
            return (p[..., 0] - a[0]) * (b[1] - a[1]) - (p[..., 1] - a[1]) * (b[0] - a[0])

        w0 = edge(p1, p2, points)
        w1 = edge(p2, p0, points)
        w2 = edge(p0, p1, points)

        # fill in only the points that are inside the triangle we are rendering
        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        ys, xs = np.where(inside)
        self.render[min_y + ys, min_x + xs] = shade_index


    # render the polyhedron faces with shading
    def render_polyhedron_faces(self, polyhedron):
        for face_index, face in enumerate(self.faces):
            # we only need 3 points to calculate the normal vector and they should be indexed in counter clockwise orientation
            p1 = polyhedron[face[0]]
            p2 = polyhedron[face[1]]
            p3 = polyhedron[face[2]]
            # calculate the normal vector and normalize it
            v1 = p2 - p1
            v2 = p3 - p1
            normal_vector = np.cross(v1, v2)
            normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)

            # check if the face is facing away
            dot_product = np.dot(normalized_normal_vector, self.camera_vector)
            if dot_product < 0:
                # calculate shading for a face
                symbol_index = np.clip(a=int(len(self.lookup_symbols) * dot_product * -1), a_min=0, a_max=len(self.lookup_symbols) - 1)
                # fill in the render matrix
                for triangle in self.triangles[face_index]:
                    self.fill_triangle(polyhedron, triangle, shade_index=symbol_index)

    def print_render(self, polyhedron):
        self.draw_method(polyhedron)
        char_matrix = self.lookup_symbols[self.render]
        self.render.fill(0)

        # Move cursor to top-left using ANSI escape code
        sys.stdout.write('\033[H\033[2J\033[3J')  # clear screen and reset cursor
        for row in char_matrix:
            sys.stdout.write(''.join(row) + '\n')
        sys.stdout.flush()


    # game loop
    def consistently_rotate_polyhedron(self, theta=np.array([0.1, 0.01, 0.05])):
        # due to rounding erros we need to change the angle of rotation of a static cube instead of actually rotating the cube
        total_theta = np.array([0.0, 0.0, 0.0])
        while(True):
            # update theta
            total_theta = (total_theta + theta) % (2 * np.pi)
            # calculate the new rotation matrix
            rotation_matrix = self.multi_dim_rotation(total_theta)
            # rotate the cube
            temp_polyhedron = self.polyhedron_offset @ rotation_matrix + self.c
            temp_polyhedron = temp_polyhedron @ self.aspect_ratio_transformation_matrix
            # calculate the render matrix based on the rotated cube and display it
            self.print_render(polyhedron=temp_polyhedron)
            # sleep
            time.sleep(0.01)


if __name__ == '__main__':
    n = len(sys.argv)
    try:
        # interpret the user input
        side_length = int(sys.argv[1])
        theta = np.array([float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])])
        shape = str(sys.argv[5])
        draw_faces = str(sys.argv[6])
    except:
        # no or incorrect user input was provided, so we use standard
        side_length = 29
        theta = np.array([0.02, 0.002, 0.001])
        shape = 'cube'
        draw_faces = 1
    
    poly = Polyhedron(shape=shape, side_length=side_length, draw_faces=draw_faces)
    poly.consistently_rotate_polyhedron(theta)
