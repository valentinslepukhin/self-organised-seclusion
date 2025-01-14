import numpy as np
import math
import matplotlib.pyplot as plt



#Here elements go

#One cavity with circle at the bottom

def rect(H, L, a, r, X, Y):
    X.append(X[-1] + a)
    Y.append(Y[-1])
    X.append(X[-1])
    Y.append(Y[-1] + H)
    x0 = X[-1] + L / 2.0
    y0 = Y[-1]

    npoints = 20
    for i in range(npoints + 1):
        phi = np.pi * (1 + i * 1.0 / npoints)
        X.append(x0 + r * np.cos(phi))
        Y.append(y0 + r * np.sin(phi))

    X.append(x0 + L / 2.0)
    Y.append(y0)
    X.append(X[-1])
    Y.append(Y[-1] - H)
    return (X, Y)



#One funnel cavity

def funnel(H1, L1, H2, L2, a, X, Y):
    X.append(X[-1] + a)
    Y.append(Y[-1])
    X.append(X[-1])
    Y.append(Y[-1] + H1)
    X.append(X[-1] - L2)
    Y.append(Y[-1] + H2)
    X.append(X[-1] + 2*L2 + L1)
    Y.append(Y[-1])
    X.append(X[-1] - L2)
    Y.append(Y[-1] - H2)
    X.append(X[-1])
    Y.append(Y[-1] - H1)
    return (X, Y)


#One hexagon column to support

def small_hex(sm_side, x0, y0):
    X = []
    Y = []
    for i in range(6):
        angle = - np.pi / 3 * i
        X.append(x0 + sm_side * np.cos(angle))
        Y.append(y0 + sm_side * np.sin(angle))
    #X.append(X[0])
    #Y.append(Y[0])
    return (X,Y)



#Hexagon inlet

def hexagon_left( params,   X, Y):
    ext2 = params['ext2']
    ext3 = params['ext3']
    ext4 = params['ext4']
    side = params['side']
    X.append(X[-1] - ext2/ np.sqrt(3))
    Y.append(Y[-1] - ext2)
    X.append(X[-1]- ext3)
    Y.append(Y[-1])
    y_center = Y[-1] + ext2 + ext4
    x_center = X[-1] + (ext2 + ext4) / np.sqrt(3) - side 
    for i in range(1,6):
        angle = - np.pi / 3 * i
        X.append(x_center + side * np.cos(angle))
        Y.append(y_center + side * np.sin(angle))
    X.append(X[-6])  # Close the hexagon
    Y.append(Y[-6] + 2*(ext2 + ext4))
    X.append(X[-1]+ ext3)
    Y.append(Y[-1])
    X.append(X[-1] + ext2/ np.sqrt(3))
    Y.append(Y[-1] - ext2)
    
    return (X, Y, x_center)


#Hexagon outlet


def hexagon_right( params,   X, Y):
    ext2 = params['ext2']
    ext3 = params['ext3']
    ext4 = params['ext4']
    side = params['side']
    X.append(X[-1] + ext2/ np.sqrt(3))
    Y.append(Y[-1] + ext2)
    X.append(X[-1] + ext3)
    Y.append(Y[-1])
    y_center = Y[-1] - ext2 - ext4
    x_center = X[-1] - (ext2 + ext4) / np.sqrt(3) + side 
    for i in range(1,6):
        angle = np.pi - np.pi / 3 * i
        X.append(x_center + side * np.cos(angle))
        Y.append(y_center + side * np.sin(angle))
    X.append(X[-6])  # Close the hexagon
    Y.append(Y[-6] - 2*(ext2 + ext4))
    X.append(X[-1]- ext3)
    Y.append(Y[-1])
    X.append(X[-1] - ext2/ np.sqrt(3))
    Y.append(Y[-1] + ext2)
    return (X, Y, x_center)


#Make a parts

#Left part (two-inlet structure)

def left_part( params):
    x0 = params['x_in']
    y0 = params['y_in']
    shift = params['shift']
    W = params['W']
    X = []
    Y = []
    X.append(x0)
    Y.append(y0 - W/2)
    X.append(X[-1] - shift)
    Y.append(Y[-1])
    X.append(X[-1] - shift )
    Y.append(Y[-1] - shift )
    X.append(X[-1] - shift)
    Y.append(Y[-1])
    X, Y, x1 = hexagon_left(params, X, Y)
    X.append(X[-1] + shift)
    Y.append(Y[-1])
    X.append(X[-1] + shift - W/2)
    Y.append(Y[-1] + shift - W/2)
    X.append(X[-1] - shift + W/2)
    Y.append(Y[-1] + shift - W/2)
    X.append(X[-1] - shift)
    Y.append(Y[-1])
    X, Y, x1 = hexagon_left(params, X, Y)
    X.append(X[-1] + shift)
    Y.append(Y[-1])
    X.append(X[-1] + shift )
    Y.append(Y[-1] - shift )
    X.append(X[-1] + shift)
    Y.append(Y[-1])
    return X,Y, x1



#Full panflute without coulumns




#Panflute with two inlets



def panflute_two_inlets(params,params_p):
    x0 = params['x_in']
    y0 = params['y_in']
    num = params['num']
    X,Y,x1 = left_part(params)
    shift = params['shift']
    h = params_p['h']
    L = params_p['L']
    r = params_p['r']
    a = params_p['a']
    H2 = params['H2']
    L2 = params['L2']
    nstart = params_p['nstart']
    nfin = params_p['nfin']
    if (num < 2):  #regular or wide
        for i in range(nstart, nfin):            
            X, Y = rect(h * (i + 1), L, a, r, X, Y)
    elif (num == 2): #exponential
        for i in range( nfin - nstart - 5):
            X, Y = rect(h * nstart*1.1**(i), L, a, r, X, Y)  
    elif (num == 3): #funnel
        for i in range(nstart, nfin):
            X, Y = funnel(h * (i + 1) - H2, L, H2, L2, a + 30, X, Y)
    X.append(X[-1] + 3 * shift)
    Y.append(Y[-1])
    X, Y, x2 = hexagon_right(params, X, Y)    
    return (X, Y, x1, x2)



#Panflute for COMSOL simulation

def panflute_comsol(params,params_p):
    X = []
    Y = []
    x0 = params['x_in']
    y0 = params['y_in']
    W = params['W']

    num = params['num']
    X.append(x0)
    Y.append(y0)
    X.append(X[-1])
    Y.append(Y[-1] + W)
    h = params_p['h']
    L = params_p['L']
    r = params_p['r']
    a = params_p['a']
    H2 = params['H2']
    L2 = params['L2']
    nstart = params_p['nstart']
    nfin = params_p['nfin']
    
    if (num < 2):  #regular or wide
        for i in range(nstart, nfin):            
            X, Y = rect(h * (i + 1), L, a, r, X, Y)
    elif (num == 2): #exponential
        for i in range( nfin - nstart - 5):
            X, Y = rect(h * nstart*1.1**(i), L, a, r, X, Y)  
    elif (num == 3): #funnel
        for i in range(nstart, nfin):
            X, Y = funnel(h * (i + 1) - H2, L, H2, L2, a + 30, X, Y) 
    X.append(X[-1] + a)
    Y.append(Y[-1])
    X.append(X[-1])
    Y.append(Y[-1] - W)
    return (X, Y)



#Helping functions to create hexagon lattice






import math

def create_hexagon(side_length, center=(1, 2)):
    """
    Create coordinates for a regular hexagon with a specified center.
    
    :param side_length: Length of each side of the hexagon
    :param center: (x0, y0) coordinates of the hexagon's center
    :return: numpy array of hexagon vertices
    """
    x0, y0 = center
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 vertices
    x = x0 + side_length * np.cos(angles)
    y = y0 + side_length * np.sin(angles)
    return np.column_stack((x, y))

def is_point_in_hexagon(point, hexagon_vertices, strict=False):
    """
    Check if a point is inside a hexagon using the ray casting algorithm.
    
    :param point: (x, y) coordinates of the point
    :param hexagon_vertices: vertices of the hexagon
    :param strict: If True, exclude points exactly on the boundary
    :return: True if point is inside, False otherwise
    """
    x, y = point
    n = len(hexagon_vertices)
    inside = False
    
    p1x, p1y = hexagon_vertices[0]
    for i in range(n + 1):
        p2x, p2y = hexagon_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    # If strict mode is on, check for exact boundary points
    if inside and strict:
        # Compute distances to each hexagon edge
        for i in range(n):
            p1 = hexagon_vertices[i]
            p2 = hexagon_vertices[(i+1) % n]
            
            # Compute distance from point to line segment
            line_vec = p2 - p1
            point_vec = np.array(point) - p1
            
            # If distance is very close to zero, it's on the boundary
            cross_product = np.abs(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)
            if np.isclose(cross_product, 0, atol=1e-10):
                # Check if point is within line segment
                dot_product = np.dot(point_vec, line_vec)
                if 0 <= dot_product <= np.dot(line_vec, line_vec):
                    return False
    
    return inside

def generate_triangular_lattice(hexagon_vertices, step=1):
    """
    Generate points for a triangular lattice inside a hexagon.
    
    :param hexagon_vertices: vertices of the hexagon
    :param step: spacing between lattice points
    :return: list of lattice points inside the hexagon
    """
    # Determine the bounding box of the hexagon
    x_min, x_max = np.min(hexagon_vertices[:,0]), np.max(hexagon_vertices[:,0])
    y_min, y_max = np.min(hexagon_vertices[:,1]), np.max(hexagon_vertices[:,1])
    
    # Generate lattice points
    lattice_points = []
    y = y_min
    alternate = False
    
    while y <= y_max:
        x = x_min
        if alternate:
            x += step / 2
        
        while x <= x_max:
            point = (x, y)
            # Use strict mode to exclude boundary points
            if is_point_in_hexagon(point, hexagon_vertices, strict=True):
                lattice_points.append(point)
            x += step
        
        y += step * math.sqrt(3) / 2
        alternate = not alternate
    
    return lattice_points




#Helping functions to create square lattice

def get_squares_in_circle(radius, x0, y0, sizex, sizey):
    bound = 60000#int(np.ceil(radius))
    # Make sure (0,0) is a grid point by using integers
    x_range = np.arange(-bound, bound + sizex, sizex) + x0 + 0.5*sizex
    y_range = np.arange(-bound, bound + sizey, sizey) + y0 + 0.5*sizey
    
    centers = []
    for x in x_range:
        for y in y_range:
            corners = [
                (x + sizex/2, y + sizey/2),
                (x + sizex/2, y - sizey/2),
                (x - sizex/2, y + sizey/2),
                (x - sizex/2, y - sizey/2)
            ]
            if all(np.sqrt((cx - x0)**2 + (cy - y0)**2) <= radius for cx, cy in corners):
                centers.append((x, y))
    
    return centers




#Outputinng to the file


def one_line(X,Y, f):
    f.write("_PLINE\n")
    for x, y in zip(X, Y):
        f.write(f"{round(x, 3)},{round(y, 3)}\n")
    f.write("C\n")
    
def inlet_columns(side, center, sm_side, xs, ys, f):
    hexagon_vertices1 = create_hexagon(side, center)
    lattice_points1 = generate_triangular_lattice(hexagon_vertices1,step  = 5*sm_side)
    lattice_x1, lattice_y1 = zip(*lattice_points1)
    for i in range(len(lattice_x1)):
        xx,yy = small_hex(sm_side, lattice_x1[i], lattice_y1[i])
        one_line(xx,yy, f)
    xx,yy = small_hex(sm_side, xs, ys )    
    one_line(xx,yy, f)



def draw_comsol_panflute( params, params_pan, f):
    X, Y = panflute_comsol(params,params_pan)
    one_line(X,Y, f)
     
    


def draw_single_panflute( params, params_pan, f):
    X, Y, x1, x2 = panflute_two_inlets(params,params_pan)
    shift = params['shift']
    sm_side = params['sm_side']
    side = params['side']
    y_in = params['y_in']
    one_line(X,Y, f)
        
    center=(x1, y_in - shift)
    xs = x1 + side + sm_side
    ys = y_in - shift
    inlet_columns(side, center, sm_side, xs, ys, f)
                
    center=(x1, y_in + shift )
    xs = x1 + side + sm_side
    ys = y_in + shift
    inlet_columns(side, center, sm_side, xs, ys, f)
        
    center=(x2, y_in )
    xs = x2 - side - sm_side
    ys = y_in
    inlet_columns(side, center, sm_side, xs, ys, f)
    
    
def draw_two_panflutes( params, params_pan, f):
    draw_single_panflute( params, params_pan, f)
    y0  = params['y_in']
    d = params['in_pair_distance']
    params['y_in'] = y0 + d
    draw_single_panflute( params, params_pan, f)
    

def draw_circle(R, x0, y0, f):
    phi = np.arange(10000)*1e-4*2*math.pi
    X = np.cos(phi)*R + x0
    Y = np.sin(phi)*R + y0
    one_line(X,Y, f)
    
def draw_one_mask(params, params_narrow, params_wide, f):
    print("num = ", params['num'])
    R = params['R']
    x0 = params['x_in']
    y0 = params['y_in']
    draw_circle(R,x0,y0, f)
    sizex = params['interpair_distance_x']
    sizey = params['interpair_distance_y']
    cen = get_squares_in_circle(R, x0, y0, sizex, sizey)
    num = params['num']
    print(len(cen))
    for c in cen:
        x_in, y_in = c
        params['x_in'] = x_in
        params['y_in'] = y_in
        if y_in < 0:
            params['num'] = num + 1
        else:
            params['num'] = num
        if (params['num'] == 0):
            draw_two_panflutes( params, params_wide, f)
        else:
            draw_two_panflutes( params, params_narrow, f)
    
    
    