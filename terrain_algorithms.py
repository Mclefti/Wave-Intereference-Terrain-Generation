import math
import random
import numpy as np

# ============================================================
#  PERLIN NOISE (2D)
# ============================================================

def make_perm(seed=0):
    rnd = random.Random(seed)
    p = list(range(256))
    rnd.shuffle(p)
    return p + p

def perlin2_factory(seed=0):
    perm = make_perm(seed)
    GRAD2 = [(math.cos(a), math.sin(a)) for a in [i * math.pi * 2 / 8 for i in range(8)]]

    def grad2(hash_val, x, y):
        g = GRAD2[hash_val & 7]
        return g[0]*x + g[1]*y

    def fade(t):
        return t*t*t*(t*(t*6 - 15) + 10)

    def lerp(a, b, t):
        return a + t*(b - a)

    def perlin2(x, y):
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u = fade(xf)
        v = fade(yf)

        aa = perm[perm[xi] + yi]
        ab = perm[perm[xi] + yi + 1]
        ba = perm[perm[xi + 1] + yi]
        bb = perm[perm[xi + 1] + yi + 1]

        x1 = lerp(grad2(aa, xf, yf), grad2(ba, xf - 1, yf), u)
        x2 = lerp(grad2(ab, xf, yf - 1), grad2(bb, xf - 1, yf - 1), u)
        return lerp(x1, x2, v)

    return perlin2


def fbm_perlin(perlin2, x, y, octaves=5, lacunarity=2.0, gain=0.5):
    amp, freq, total, norm = 1.0, 1.0, 0.0, 0.0
    for _ in range(octaves):
        total += amp * perlin2(x * freq, y * freq)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


# ============================================================
#  SIMPLEX NOISE (2D)
# ============================================================

def simplex2_factory(seed=0):
    rnd = random.Random(seed)
    p = list(range(256))
    rnd.shuffle(p)
    perm = p + p
    grad3 = [(1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),
             (1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1),
             (0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1)]
    F2 = 0.5*(math.sqrt(3.0)-1.0)
    G2 = (3.0 - math.sqrt(3.0))/6.0

    def dot(g, x, y): return g[0]*x + g[1]*y

    def simplex2(xin, yin):
        s = (xin + yin) * F2
        i = math.floor(xin + s)
        j = math.floor(yin + s)
        t = (i + j) * G2
        X0 = i - t
        Y0 = j - t
        x0 = xin - X0
        y0 = yin - Y0

        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2

        ii = int(i) & 255
        jj = int(j) & 255
        gi0 = perm[ii + perm[jj]] % 12
        gi1 = perm[ii + i1 + perm[jj + j1]] % 12
        gi2 = perm[ii + 1 + perm[jj + 1]] % 12

        n0 = n1 = n2 = 0.0
        t0 = 0.5 - x0*x0 - y0*y0
        if t0 > 0:
            t0 *= t0
            n0 = t0 * t0 * dot(grad3[gi0], x0, y0)
        t1 = 0.5 - x1*x1 - y1*y1
        if t1 > 0:
            t1 *= t1
            n1 = t1 * t1 * dot(grad3[gi1], x1, y1)
        t2 = 0.5 - x2*x2 - y2*y2
        if t2 > 0:
            t2 *= t2
            n2 = t2 * t2 * dot(grad3[gi2], x2, y2)

        return 70.0 * (n0 + n1 + n2)

    return simplex2


# ============================================================
#  DIAMOND–SQUARE (FRACTAL SUBDIVISION)
# ============================================================

def diamond_square(n, scale=1.0, roughness=0.6, seed=0):
    rnd = random.Random(seed)
    size = 2**n + 1
    a = [[0.0]*size for _ in range(size)]
    a[0][0] = rnd.random()
    a[0][-1] = rnd.random()
    a[-1][0] = rnd.random()
    a[-1][-1] = rnd.random()
    step = size - 1

    while step > 1:
        half = step // 2
        for x in range(0, size-1, step):
            for y in range(0, size-1, step):
                avg = (a[x][y] + a[x+step][y] + a[x][y+step] + a[x+step][y+step]) / 4.0
                a[x+half][y+half] = avg + (rnd.random()*2 - 1)*scale
        for x in range(0, size, half):
            for y in range((x+half)%step, size, step):
                s = []
                if x-half >= 0: s.append(a[x-half][y])
                if x+half < size: s.append(a[x+half][y])
                if y-half >= 0: s.append(a[x][y-half])
                if y+half < size: s.append(a[x][y+half])
                a[x][y] = sum(s)/len(s) + (rnd.random()*2 - 1)*scale
        step //= 2
        scale *= roughness

    return np.array(a)


# ============================================================
#  GAUSSIAN PROCESS FIELD (SPECTRAL SAMPLING)
# ============================================================

def spectral_gp_2d(shape=(128,128), length_scale=10.0, variance=1.0, seed=0):
    rnd = np.random.RandomState(seed)
    nx, ny = shape
    kx = np.fft.fftfreq(nx)[:,None]
    ky = np.fft.fftfreq(ny)[None,:]
    k2 = (kx**2 + ky**2)
    power = variance * np.exp(-2*(np.pi**2) * (length_scale**2) * k2)
    coeff = (rnd.normal(size=shape) + 1j*rnd.normal(size=shape)) * np.sqrt(power)
    coeff = (coeff + np.conj(np.flipud(np.fliplr(coeff))))/2
    field = np.fft.ifft2(coeff).real
    return (field - field.min()) / (field.max() - field.min())


# ============================================================
#  MULTISCALE (WAVELET-LIKE) SYNTHESIS
# ============================================================

def multiscale_upsampled_field(nx=128, ny=128, levels=6, H=0.7, seed=0):
    rnd = np.random.RandomState(seed)
    field = np.zeros((ny, nx))
    for j in range(levels):
        sx, sy = max(1, nx // (2**j)), max(1, ny // (2**j))
        noise = rnd.normal(size=(sy, sx))
        up = np.kron(noise, np.ones((2**j, 2**j)))[:ny, :nx]
        sigma = 2**(-j * H)
        field += sigma * up
    return (field - field.min()) / (field.max() - field.min())


# ============================================================
#  WAVE INTERFERENCE
# ============================================================

def interference_grid(sources, grid_x, grid_y, k=2*np.pi/5.0, damping=0.0):
    Nx, Ny = len(grid_x), len(grid_y)
    X, Y = np.meshgrid(grid_x, grid_y, indexing='xy')
    field = np.zeros_like(X)
    for (sx, sy, A, phi) in sources:
        r = np.hypot(X - sx, Y - sy) + 1e-8
        field += (A / np.sqrt(r)) * np.cos(k * r + phi) * np.exp(-damping * r)
    return (field - field.min()) / (field.max() - field.min())

def random_interference_grid(n_sources=5, grid_range=10.0, grid_size=256, seed=0):
    rnd = np.random.RandomState(seed)
    grid_x = np.linspace(-grid_range, grid_range, grid_size)
    grid_y = np.linspace(-grid_range, grid_range, grid_size)
    sources = []
    for _ in range(n_sources):
        sx = rnd.uniform(-grid_range/2, grid_range/2)
        sy = rnd.uniform(-grid_range/2, grid_range/2)
        A = rnd.uniform(0.5, 1.5)
        k = 2 * np.pi / rnd.uniform(2.0, 8.0)
        phi = rnd.uniform(0, 2*np.pi)
        sources.append((sx, sy, A, phi, k))

    X, Y = np.meshgrid(grid_x, grid_y, indexing='xy')
    field = np.zeros_like(X)
    for (sx, sy, A, phi, k) in sources:
        r = np.hypot(X - sx, Y - sy) + 1e-8
        field += (A / np.sqrt(r)) * np.cos(k * r + phi)
    return (field - field.min()) / (field.max() - field.min())


# ============================================================
#  TERRAIN FIELD COMBINER
# ============================================================

def combine_fields(field1, field2, method='blend', weight1=0.5, weight2=0.5, normalize=True):
    if field1.shape != field2.shape:
        raise ValueError("Fields must have the same shape")

    if method == 'blend':
        combined = weight1 * field1 + weight2 * field2
    elif method == 'add':
        combined = field1 * weight1 + field2 * weight2
    elif method == 'multiply':
        combined = field1 * field2
    elif method == 'subtract':
        combined = field1 - field2
    elif method == 'max':
        combined = np.maximum(field1, field2)
    elif method == 'min':
        combined = np.minimum(field1, field2)
    elif method == 'ridge':
        combined = 1.0 - np.abs(field1 - field2)
    else:
        raise ValueError(f"Unknown method: {method}")

    if normalize:
        cmin, cmax = combined.min(), combined.max()
        if cmax - cmin > 1e-12:
            combined = (combined - cmin) / (cmax - cmin)
        else:
            combined = np.zeros_like(combined)
    return combined



# ============================================================
#  EXPORT HEIGHTMAP TO .OBJ MESH
# ============================================================

def save_heightmap_as_obj(heightmap, filename, scale=(1.0, 1.0, 1.0)):
    """
    Convert a 2D heightmap (NumPy array) into a 3D mesh and save as .obj.

    Parameters
    ----------
    heightmap : np.ndarray
        2D array of height values in [0, 1].
    filename : str
        Output .obj file path.
    scale : (float, float, float)
        Scaling factors for (x, y, z). Example: (1.0, 1.0, 20.0) to exaggerate height.

    Notes
    -----
    - Exports in Wavefront OBJ format.
    - Works perfectly for Blender, Unity, Unreal, etc.
    """
    h, w = heightmap.shape
    sx, sy, sz = scale

    with open(filename, "w") as f:
        f.write("# Generated by terrain_algorithms.py\n")

        # Write vertices
        for y in range(h):
            for x in range(w):
                z = heightmap[y, x] * sz
                f.write(f"v {x * sx:.6f} {y * sy:.6f} {z:.6f}\n")

        # Write faces (two triangles per grid cell)
        def vid(x, y):
            return y * w + x + 1  # OBJ indices start at 1

        for y in range(h - 1):
            for x in range(w - 1):
                v1 = vid(x, y)
                v2 = vid(x + 1, y)
                v3 = vid(x + 1, y + 1)
                v4 = vid(x, y + 1)
                f.write(f"f {v1} {v2} {v3}\n")
                f.write(f"f {v1} {v3} {v4}\n")

    print(f"[✓] OBJ saved to {filename} ({w}x{h} vertices)")
