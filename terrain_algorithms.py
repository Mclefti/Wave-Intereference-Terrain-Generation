# terrain_algorithms.py
import numpy as np
import math, random

# ===============================================================
# === Noise & Terrain Algorithms ================================
# ===============================================================

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

    def fade(t): return t*t*t*(t*(t*6 - 15) + 10)
    def lerp(a, b, t): return a + t*(b - a)

    def perlin2(x, y):
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u, v = fade(xf), fade(yf)
        aa = perm[perm[xi] + yi]
        ab = perm[perm[xi] + yi + 1]
        ba = perm[perm[xi + 1] + yi]
        bb = perm[perm[xi + 1] + yi + 1]
        x1 = lerp(grad2(aa, xf, yf), grad2(ba, xf - 1, yf), u)
        x2 = lerp(grad2(ab, xf, yf - 1), grad2(bb, xf - 1, yf - 1), u)
        return lerp(x1, x2, v)
    return perlin2

def fbm_perlin_sample(perlin2, x, y, octaves=5, lacunarity=2.0, gain=0.5):
    amp, freq, total, norm = 1.0, 1.0, 0.0, 0.0
    for _ in range(octaves):
        total += amp * perlin2(x * freq, y * freq)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm

def perlin_noise(size=256, scale=10.0, seed=0, octaves=5):
    perlin = perlin2_factory(seed)
    xs = np.linspace(0, scale, size)
    ys = np.linspace(0, scale, size)
    field = np.zeros((size, size), dtype=np.float64)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            field[j, i] = fbm_perlin_sample(perlin, x, y, octaves=octaves)
    field = (field - field.min()) / (field.max() - field.min())
    return field

def diamond_square(n=256, roughness=0.5, seed=0):
    np.random.seed(seed)
    size = n
    data = np.zeros((size+1, size+1))
    data[0,0] = np.random.rand()
    data[0,size] = np.random.rand()
    data[size,0] = np.random.rand()
    data[size,size] = np.random.rand()
    step = size
    while step > 1:
        half = step // 2
        for x in range(half, size, step):
            for y in range(half, size, step):
                avg = (data[x-half,y-half]+data[x-half,y+half]+data[x+half,y-half]+data[x+half,y+half])/4.0
                data[x,y] = avg + (np.random.rand()-0.5)*roughness*step/size
        for x in range(0, size+1, half):
            for y in range((x+half)%step, size+1, step):
                s=[]
                if x-half>=0: s.append(data[x-half,y])
                if x+half<=size: s.append(data[x+half,y])
                if y-half>=0: s.append(data[x,y-half])
                if y+half<=size: s.append(data[x,y+half])
                avg=np.mean(s)
                data[x,y]=avg+(np.random.rand()-0.5)*roughness*step/size
        step//=2
    data=(data-data.min())/(data.max()-data.min())
    return data

def spectral_gaussian_process(size=256, H=0.7, seed=0):
    np.random.seed(seed)
    freqs = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freqs, freqs)
    radius = np.sqrt(fx**2 + fy**2)
    radius[0,0] = 1.0
    spectrum = radius ** (-(H + 1.0))
    phase = np.random.rand(size, size) * 2 * np.pi
    real = np.cos(phase) * spectrum
    imag = np.sin(phase) * spectrum
    field = np.fft.ifft2(real + 1j * imag).real
    field = (field - field.min()) / (field.max() - field.min())
    return field

def random_interference_grid(n_sources=8, grid_range=10, grid_size=256, seed=0, circular=False):
    np.random.seed(seed)
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for _ in range(n_sources):
        amp = np.random.uniform(0.2, 1.0)
        freq = np.random.uniform(0.5, 2.5)
        phase = np.random.uniform(0, 2*np.pi)
        if circular:
            cx, cy = np.random.uniform(-grid_range, grid_range, 2)
            r = np.sqrt((X - cx)**2 + (Y - cy)**2)
            Z += amp * np.sin(freq * r + phase)
        else:
            direction = np.random.uniform(0, 2*np.pi)
            kx, ky = np.cos(direction) * freq, np.sin(direction) * freq
            Z += amp * np.sin(kx * X + ky * Y + phase)
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return Z

# --- Harmonic Base Field ---
def harmonic_field(size=256, frequency=3.0, seed=0):
    np.random.seed(seed)
    x = np.linspace(0, 2*np.pi, size)
    y = np.linspace(0, 2*np.pi, size)
    X, Y = np.meshgrid(x, y)
    phase = np.random.rand() * 2 * np.pi
    field = np.sin(frequency * X + phase) + np.cos(frequency * Y + phase)
    field = (field - field.min()) / (field.max() - field.min())
    return field

# ===============================================================
# === Combine + Export ==========================================
# ===============================================================

def combine_fields(field1, field2, method="blend", weight1=0.6, weight2=0.4):
    field1 = (field1 - field1.min()) / (field1.max() - field1.min())
    field2 = (field2 - field2.min()) / (field2.max() - field2.min())
    if method == "blend":
        f = weight1 * field1 + weight2 * field2
    elif method == "ridge":
        f = 1.0 - np.abs(field1 - field2)
    else:
        f = field1
    f = (f - f.min()) / (f.max() - f.min())
    return f

def export_obj(field, filename="terrain.obj", scale_z=50.0):
    h, w = field.shape
    with open(filename, "w") as f:
        for y in range(h):
            for x in range(w):
                z = field[y, x] * scale_z
                f.write(f"v {x} {y} {z}\n")
        for y in range(h-1):
            for x in range(w-1):
                v1 = y * w + x + 1
                v2 = v1 + 1
                v3 = v1 + w
                v4 = v3 + 1
                f.write(f"f {v1} {v2} {v4}\n")
                f.write(f"f {v1} {v4} {v3}\n")
