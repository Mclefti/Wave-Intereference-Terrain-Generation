import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math, random

# ===============================================================
# === Algorithms ================================================
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

# ===============================================================
# === UI ========================================================
# ===============================================================

class TerrainUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Field Terrain Generator + OBJ Export + Wave Sources + Labels")
        self.root.configure(bg="#222")

        self.field1 = None
        self.field2 = None
        self.hybrid = None

        self.size = tk.IntVar(value=256)
        self.seed1 = tk.IntVar(value=42)
        self.seed2 = tk.IntVar(value=99)
        self.wave1_circular = tk.BooleanVar(value=False)
        self.wave2_circular = tk.BooleanVar(value=False)
        self.sources1 = tk.IntVar(value=8)
        self.sources2 = tk.IntVar(value=8)
        self.blend_weight = tk.DoubleVar(value=0.6)
        self.hybrid_mode = tk.StringVar(value="blend")

        self._build_ui()

    def _build_ui(self):
        ctrl = tk.Frame(self.root, bg="#333", padx=10, pady=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        # --- Field 1 ---
        tk.Label(ctrl, text="Field 1 (Base)", bg="#333", fg="white", font=("Consolas", 10, "bold")).pack()
        self.method1 = ttk.Combobox(ctrl, values=["Perlin", "DiamondSquare", "SpectralGP", "WaveInterference"], state="readonly")
        self.method1.current(0)
        self.method1.pack(pady=2)
        tk.Entry(ctrl, textvariable=self.seed1).pack(pady=2)
        ttk.Checkbutton(ctrl, text="Circular Waves", variable=self.wave1_circular).pack()
        tk.Label(ctrl, text="Wave Sources").pack()
        tk.Entry(ctrl, textvariable=self.sources1).pack(pady=2)
        ttk.Button(ctrl, text="Generate Field 1", command=self.gen_field1).pack(pady=4)

        # --- Field 2 ---
        tk.Label(ctrl, text="Field 2 (Modifier)", bg="#333", fg="white", font=("Consolas", 10, "bold")).pack(pady=(10,0))
        self.method2 = ttk.Combobox(ctrl, values=["Perlin", "DiamondSquare", "SpectralGP", "WaveInterference"], state="readonly")
        self.method2.current(3)
        self.method2.pack(pady=2)
        tk.Entry(ctrl, textvariable=self.seed2).pack(pady=2)
        ttk.Checkbutton(ctrl, text="Circular Waves", variable=self.wave2_circular).pack()
        tk.Label(ctrl, text="Wave Sources").pack()
        tk.Entry(ctrl, textvariable=self.sources2).pack(pady=2)
        ttk.Button(ctrl, text="Generate Field 2", command=self.gen_field2).pack(pady=4)

        # --- Combine ---
        tk.Label(ctrl, text="Combination", bg="#333", fg="white", font=("Consolas", 10, "bold")).pack(pady=(10,0))
        ttk.Combobox(ctrl, textvariable=self.hybrid_mode, values=["blend", "ridge"], state="readonly").pack(pady=2)
        ttk.Scale(ctrl, from_=0, to=1, orient="horizontal", variable=self.blend_weight).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Combine Fields", command=self.combine_fields_ui).pack(pady=5)

        # --- Export ---
        ttk.Button(ctrl, text="Export Hybrid as OBJ", command=self.export_current_obj).pack(pady=8)

        # --- Misc ---
        tk.Label(ctrl, text="Grid Size", bg="#333", fg="white").pack(pady=(10,0))
        tk.Entry(ctrl, textvariable=self.size).pack(pady=2)
        ttk.Button(ctrl, text="Quit", command=self.root.destroy).pack(pady=10)

        self.status_label = ttk.Label(ctrl, text="Ready.", background="#333", foreground="#ccc")
        self.status_label.pack(fill=tk.X, pady=5)

        self.fig, self.ax = plt.subplots(1,3,figsize=(8,3))
        for a in self.ax: a.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _generate(self, method, seed, n, circular, n_sources):
        if method == "Perlin":
            return perlin_noise(size=n, scale=20.0, seed=seed)
        elif method == "DiamondSquare":
            return diamond_square(n=n, seed=seed)
        elif method == "SpectralGP":
            return spectral_gaussian_process(size=n, seed=seed)
        elif method == "WaveInterference":
            return random_interference_grid(grid_size=n, seed=seed, circular=circular, n_sources=n_sources)
        else:
            return None

    def gen_field1(self):
        n, s, m, c, ns = self.size.get(), self.seed1.get(), self.method1.get(), self.wave1_circular.get(), self.sources1.get()
        self.field1 = self._generate(m, s, n, c, ns)
        self.status_label.config(text=f"Generated Field 1: {m} (Sources={ns})")
        self._update_plot()

    def gen_field2(self):
        n, s, m, c, ns = self.size.get(), self.seed2.get(), self.method2.get(), self.wave2_circular.get(), self.sources2.get()
        self.field2 = self._generate(m, s, n, c, ns)
        self.status_label.config(text=f"Generated Field 2: {m} (Sources={ns})")
        self._update_plot()

    def combine_fields_ui(self):
        if self.field1 is None or self.field2 is None:
            self.status_label.config(text="Please generate both fields first.")
            return
        mode = self.hybrid_mode.get()
        w1, w2 = self.blend_weight.get(), 1 - self.blend_weight.get()
        self.hybrid = combine_fields(self.field1, self.field2, method=mode, weight1=w1, weight2=w2)
        self.status_label.config(text=f"Combined fields with mode={mode}")
        self._update_plot()

    def export_current_obj(self):
        if self.hybrid is None:
            self.status_label.config(text="Generate a hybrid first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".obj", filetypes=[("OBJ Files", "*.obj")])
        if not path:
            return
        export_obj(self.hybrid, path, scale_z=50.0)
        self.status_label.config(text=f"Exported OBJ: {path}")

    def _update_plot(self):
        self.fig.clf()
        cols, titles, details = [], [], []

        if self.field1 is not None:
            algo1 = self.method1.get()
            cols.append(self.field1)
            titles.append(f"Field 1 – {algo1}")
            if algo1 == "WaveInterference":
                details.append("Bright = constructive, dark = destructive")
            else:
                details.append("Brown = high, green = low")

        if self.field2 is not None:
            algo2 = self.method2.get()
            cols.append(self.field2)
            titles.append(f"Field 2 – {algo2}")
            if algo2 == "WaveInterference":
                details.append("Bright = constructive, dark = destructive")
            else:
                details.append("Brown = high, green = low")

        if self.hybrid is not None:
            mode = self.hybrid_mode.get()
            cols.append(self.hybrid)
            titles.append(f"Result (Hybrid Mode: {mode})")
            details.append("Brown = high, green = low")

        # Create axes and show images
        for i, field in enumerate(cols):
            ax = self.fig.add_subplot(1, len(cols), i + 1)
            ax.imshow(field, cmap="terrain", origin="lower")
            ax.axis("off")

            # Title inside but near the top (visible area)
            ax.text(
                0.5, 1.05, titles[i],
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color="white",
                bbox=dict(facecolor='black', alpha=0.4, pad=3, edgecolor='none')
            )

            # Field-specific color description below
            ax.text(
                0.5, -0.10, details[i],
                transform=ax.transAxes,
                ha="center", va="top",
                color="#ddd", fontsize=8
            )

        # Global legend text
        self.fig.text(
            0.5, -0.08,
            "Color Key: dark green = lowlands, light green/yellow = midlands, brown = hills, white = peaks",
            ha="center", color="#aaa", fontsize=8
        )

        # Adjust margins so titles fit inside visible area
        self.fig.subplots_adjust(top=0.9, bottom=0.2, wspace=0.05)
        self.canvas.draw()

# ===============================================================
# === Run =======================================================
# ===============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = TerrainUI(root)
    root.mainloop()
