import numpy as np
import math
import time
import sys
import threading
import queue
import trimesh

class LoadPolyhedron:
    def load_polyhedron_file(self, filepath, lower, higher):
        mesh = trimesh.load(filepath, force='mesh')
        init_vertices = np.array([np.array(lst) for lst in mesh.vertices])
        polys = mesh.faces
        init_normals = np.array([np.array(lst) for lst in mesh.vertex_normals])

        vertices, inverse_indices_vertices = np.unique(init_vertices, axis=0, return_inverse=True)
        vertices_lookup = dict(enumerate(inverse_indices_vertices))

        normals, inverse_indices_normals = np.unique(init_normals, axis=0, return_inverse=True)
        normals_lookup = dict(enumerate(inverse_indices_normals))

        
        vertices = np.vstack(vertices)

        # Compute center and overall scale
        min_vals = vertices.min(axis=0)
        max_vals = vertices.max(axis=0)
        center = (min_vals + max_vals) / 2
        scale = (max_vals - min_vals).max()  # Uniform scale

        # Map to [lower, higher] preserving proportions
        normalized_size = higher - lower
        vertices = (vertices - center) / scale * normalized_size + (lower + higher) / 2
        
        faces = []
        edges = []
        for face in polys:
            # find triangles
            faces.extend(self._triangulate_convex_polygon(face))
            # find edges
            for i in range(len(face)):
                edges.append((face[i], face[(i + 1) % len(face)]))
        faces = np.array(faces)
        edges = np.array(edges)

        return vertices, normals, vertices_lookup, normals_lookup, faces, edges

    
    # fan triangulation of a convex polygon
    def _triangulate_convex_polygon(self, face):

        triangles = []
        n = len(face)
        if n < 3:
            raise ValueError("Polygon has less than 3 vertices.")
        for i in range(1, n - 1):
            # pivot is vertex with index 0
            triangles.append((face[0], face[i], face[i + 1]))
        return triangles


class Polyhedron:
    def __init__(self, filepath='cube.obj', max_height=29, aspect_ratio=1.67, draw_faces=False):
        self.__draw_faces = draw_faces
        self.__camera_vector = np.array([0, 0, -1])

        self.__lookup_symbols = np.array([' ', '-', '~', ':', ';', '!', '+', '<', '?', '/', '|', '*', 'O', '$', '%', '#', '@'])
        self.__lookup_black = len(self.__lookup_symbols) - 1

        self.__aspect_ratio = aspect_ratio
        self.__aspect_ratio_transformation_matrix = np.array([[self.__aspect_ratio, 0, 0], [0, 1, 0], [0, 0, 1]])

        self.__filepath = filepath
        self.__max_height = max_height
        self.__polyhedron, self.__normals, self.__polyhedron_lookup, self.__normals_lookup, self.__render_buffer, self.__depth_buffer, self.__buffer_pixel_lock, self.__faces, self.__edges = self._generate_polyhedron_and_render()
        
        self.__c = np.mean(self.__polyhedron, axis=0)
        self.__polyhedron_offset = self.__polyhedron - self.__c

        self.__thread_pool = []

        if draw_faces: 
            self.__draw_method = self._render_polyhedron_faces

            self.__task_queue = queue.Queue()
            self.__shutdown_flag = threading.Event()
            for i in range(16):
                self._add_thread()

        else:
            
            edge_indices = np.array([[self.__polyhedron_lookup[a], self.__polyhedron_lookup[b]] for a, b in self.__edges])

            # Get the corresponding vertex positions
            v_start = self.__polyhedron[edge_indices[:, 0]]  
            v_end = self.__polyhedron[edge_indices[:, 1]]    

            # Compute edge vectors and their lengths
            edge_vectors = v_end - v_start
            edge_lengths = np.linalg.norm(edge_vectors, axis=1)

            # Maximum edge length
            side_length = edge_lengths.max()

            self.__draw_method = self._render_polyhedron_edges
            self.__density_lookup = {}
            self.__num_sections = 64
            for i in range(self.__num_sections):
                angle = i / (self.__num_sections - 1)
                density_modifier = 1 + (self.__aspect_ratio - 1) * angle
                density = np.linspace(0, 1, num=int(side_length * density_modifier)).reshape(-1, 1)
                self.__density_lookup[i] = density


    def _add_thread(self):
        t = threading.Thread(target=self.__worker_loop, daemon=True)
        t.start()
        self.__thread_pool.append(t)


    # generates the rotation matrix for the x element of theta
    def _x_rotation(self, theta_x):
        cos = np.cos(theta_x)
        sin = np.sin(theta_x)
        x_rotation_matrix = np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        return x_rotation_matrix


    # generates the rotation matrix for the y element of theta
    def _y_rotation(self, theta_y):
        cos = np.cos(theta_y)
        sin = np.sin(theta_y)
        y_rotation_matrix = np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        return y_rotation_matrix


    # generates the rotation matrix for the z element of theta
    def _z_rotation(self, theta_z):
        cos = np.cos(theta_z)
        sin = np.sin(theta_z)
        z_rotation_matrix = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        return z_rotation_matrix


    # calculates a full rotation matrix for a 3 dimensional rotation
    def _multi_dim_rotation(self, theta_xyz):
        rotation_matrix = self._x_rotation(theta_xyz[0])
        if theta_xyz[1] != 0:
            rotation_matrix = rotation_matrix @ self._y_rotation(theta_xyz[1])
        if theta_xyz[2] != 0:
            rotation_matrix = rotation_matrix @ self._z_rotation(theta_xyz[2])
        return rotation_matrix


    # generates a general polyhedron and instantiates the render matrix
    def _generate_polyhedron_and_render(self):
        render_dim = (math.ceil(self.__max_height * np.sqrt(3)), math.ceil(self.__max_height * np.sqrt(3)))
        dist_to_center = self.__max_height // 2
        lower = render_dim[0] // 2 - dist_to_center
        higher = render_dim[0] // 2 + dist_to_center
        # generate the polyhedron
        load = LoadPolyhedron()
        polyhedron, normals, polyhedron_lookup, normals_lookup, faces, edges = load.load_polyhedron_file(filepath=self.__filepath, lower=lower, higher=higher)

        # instantiate the render matrix
        width = math.ceil(render_dim[0] * self.__aspect_ratio)
        height = render_dim[1]
        render_buffer = np.zeros((height, width), dtype=int)
        depth_buffer = render_buffer.copy()

        # instantiate lock for pixels
        buffer_pixel_lock = np.array([[threading.Lock() for _ in range(width)] for _ in range(height)])

        return polyhedron, normals, polyhedron_lookup, normals_lookup, render_buffer, depth_buffer, buffer_pixel_lock, faces, edges


    # renders the polyhedron as a wire frame
    # ONLY SAFE FOR SINGLE THREADED RENDERING
    def _render_polyhedron_edges(self):
        # iterate over the edges of the polyhedron
        for endpoint_0, endpoint_1 in self.__edges:
            # we calculate a new density modifier for the lines so that vertical lines have less density and horizontal lines have more density because of the aspect ratio of monospace font
            dx = np.abs(self.__temp_polyhedron[self.__polyhedron_lookup[endpoint_1]][0] - self.__temp_polyhedron[self.__polyhedron_lookup[endpoint_0]][0])
            dy = np.abs(self.__temp_polyhedron[self.__polyhedron_lookup[endpoint_1]][1] - self.__temp_polyhedron[self.__polyhedron_lookup[endpoint_0]][1])
            angle = 2 * np.abs(np.arctan2(dy, dx)) / np.pi
            inverted_angle = 1 - angle
            angle_section_index = np.clip(np.round(inverted_angle * self.__num_sections).astype(int), 0, self.__num_sections - 1)
            # lookup closest density in lookup table
            density = self.__density_lookup[angle_section_index]

            # calculate points on the edge based on the calculated density
            points = np.round(self.__temp_polyhedron[self.__polyhedron_lookup[endpoint_0]] + density * (self.__temp_polyhedron[self.__polyhedron_lookup[endpoint_1]] - self.__temp_polyhedron[self.__polyhedron_lookup[endpoint_0]])).astype(int)
            # draw the indexes of the symbols in the render matrix
            self.__render_buffer[np.clip(points[:, 1], 0, self.__render_buffer.shape[0] - 1), np.clip(points[:, 0], 0, self.__render_buffer.shape[1] - 1)] = self.__lookup_black


    def __worker_loop(self):
            while not self.__shutdown_flag.is_set():
                try:
                    triangle = self.__task_queue.get(timeout=0.1)
                    try:
                        self._fill_triangle(triangle)
                    finally:
                        self.__task_queue.task_done()
                except queue.Empty:
                    continue


    # rendering the triangle
    def _fill_triangle(self, triangle):
        # for calculating which side of the line points are on
        def edge(a, b, p):
            return (p[..., 0] - a[0]) * (b[1] - a[1]) - (p[..., 1] - a[1]) * (b[0] - a[0])

        # bounding box for triangle
        p0 = self.__temp_polyhedron[self.__polyhedron_lookup[triangle[0]]]
        p1 = self.__temp_polyhedron[self.__polyhedron_lookup[triangle[1]]]
        p2 = self.__temp_polyhedron[self.__polyhedron_lookup[triangle[2]]]
        min_x = int(max(min(p0[0], p1[0], p2[0]), 0))
        max_x = int(min(max(p0[0], p1[0], p2[0]), self.__render_buffer.shape[1] - 1))
        min_y = int(max(min(p0[1], p1[1], p2[1]), 0))
        max_y = int(min(max(p0[1], p1[1], p2[1]), self.__render_buffer.shape[0] - 1))

        # normals
        n0 = self.__temp_normals[self.__normals_lookup[triangle[0]]]
        n1 = self.__temp_normals[self.__normals_lookup[triangle[1]]]
        n2 = self.__temp_normals[self.__normals_lookup[triangle[2]]]

        area = edge(p0, p1, p2)
        if area == 0:
            return

        # generate a mesh grid
        xs, ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        points = np.stack((xs, ys), axis=-1)

        w0 = edge(p1, p2, points) / area
        w1 = edge(p2, p0, points) / area
        w2 = edge(p0, p1, points) / area

        # fill in only the points that are inside the triangle we are rendering
        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0))
        ys, xs = np.where(inside)

        for dy, dx, alpha, beta, gamma in zip(ys, xs, w0[inside], w1[inside], w2[inside]):
            interpolated_normal = alpha * n0 + beta * n1 + gamma * n2
            interpolated_normal /= np.linalg.norm(interpolated_normal)
            interpolated_z = alpha * p0[2] + beta * p1[2] + gamma * p2[2]
            
            y_index = min_y + dy
            x_index = min_x + dx
            if self.__depth_buffer[y_index, x_index] < interpolated_z:
                shade_index = np.clip(a=int(len(self.__lookup_symbols) * -np.dot(interpolated_normal, self.__camera_vector)), a_min=1, a_max=len(self.__lookup_symbols) - 1)
                lock = self.__buffer_pixel_lock[y_index, x_index]
                with lock:
                    self.__render_buffer[y_index, x_index] = shade_index
                    self.__depth_buffer[y_index, x_index] = interpolated_z


    # render the polyhedron faces with shading
    def _render_polyhedron_faces(self):
        for index, face in enumerate(self.__faces):
            normal_vector = np.mean([self.__temp_normals[self.__normals_lookup[face[0]]], self.__temp_normals[self.__normals_lookup[face[1]]], self.__temp_normals[self.__normals_lookup[face[2]]]], axis=0)

            # check if the face is facing away
            dot_product = np.dot(normal_vector, self.__camera_vector)
            if dot_product < 0:
                # fill in the render matrix
                self.__task_queue.put(face)


    def _print_render(self):
        self.__draw_method()

        # wait for all threads to finish
        if self.__draw_faces:
            self.__task_queue.join()


        char_matrix = self.__lookup_symbols[self.__render_buffer]
        self.__render_buffer.fill(0)
        self.__depth_buffer.fill(0)

        # Move cursor to top-left using ANSI escape code
        output = '\033[H\033[2J\033[3J' + '\n'.join(''.join(row).rstrip() for row in char_matrix) + '\n' # move cursor to overwrite screen and create string representation
        sys.stdout.write(output)
        sys.stdout.flush()


    # game loop
    def consistently_rotate_polyhedron(self, theta=np.array([0.1, 0.01, 0.05])):
        total_theta = np.array([0.0, 0.0, 0.0])
        try:
            while(True):
                # update theta
                total_theta = (total_theta + theta) % (2 * np.pi)
                # calculate the new rotation matrix
                rotation_matrix = self._multi_dim_rotation(total_theta)
                # rotate the cube
                self.__temp_polyhedron = self.__polyhedron_offset @ rotation_matrix + self.__c
                self.__temp_polyhedron = self.__temp_polyhedron @ self.__aspect_ratio_transformation_matrix
                # rotate normals
                self.__temp_normals = self.__normals @ rotation_matrix
                # calculate the render matrix based on the rotated cube and display it
                self._print_render()

                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Keyboard interrupt received.")
        finally:
            print("Shutting down...")
            self.__shutdown_flag.set()
            for t in self.__thread_pool:
                t.join()
            print("Gracefully shut down.")
            if __name__ == "__main__":
                sys.exit(0)


if __name__ == '__main__':
    n = len(sys.argv)
    try:
        # interpret the user input
        draw_faces = bool(int(sys.argv[1]))
        max_height = int(sys.argv[2])
        theta = np.array([float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])])
        filepath = str(sys.argv[6])
    except:
        # no or incorrect user input was provided, so we use standard
        max_height = 29
        theta = np.array([0.02, 0.002, 0.001])
        filepath = 'toroidal_polyhedron.obj'
        draw_faces = 1

    poly = Polyhedron(filepath=filepath, max_height=max_height, draw_faces=draw_faces)
    poly.consistently_rotate_polyhedron(theta)
