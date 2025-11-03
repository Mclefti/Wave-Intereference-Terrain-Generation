# terrain_ui.py
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from terrain_algorithms import *

class TerrainUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Field Terrain Generator + Harmonic Base + OBJ Export")
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

        methods = ["Perlin", "DiamondSquare", "SpectralGP", "WaveInterference", "Harmonic"]

        # --- Field 1 ---
        tk.Label(ctrl, text="Field 1 (Base)", bg="#333", fg="white", font=("Consolas", 10, "bold")).pack()
        self.method1 = ttk.Combobox(ctrl, values=methods, state="readonly")
        self.method1.current(0)
        self.method1.pack(pady=2)
        tk.Entry(ctrl, textvariable=self.seed1).pack(pady=2)
        ttk.Checkbutton(ctrl, text="Circular Waves", variable=self.wave1_circular).pack()
        tk.Label(ctrl, text="Wave Sources").pack()
        tk.Entry(ctrl, textvariable=self.sources1).pack(pady=2)
        ttk.Button(ctrl, text="Generate Field 1", command=self.gen_field1).pack(pady=4)

        # --- Field 2 ---
        tk.Label(ctrl, text="Field 2 (Modifier)", bg="#333", fg="white", font=("Consolas", 10, "bold")).pack(pady=(10,0))
        self.method2 = ttk.Combobox(ctrl, values=methods, state="readonly")
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
        elif method == "Harmonic":
            return harmonic_field(size=n, seed=seed)
        else:
            return None

    def gen_field1(self):
        n, s, m, c, ns = self.size.get(), self.seed1.get(), self.method1.get(), self.wave1_circular.get(), self.sources1.get()
        self.field1 = self._generate(m, s, n, c, ns)
        self.status_label.config(text=f"Generated Field 1: {m}")
        self._update_plot()

    def gen_field2(self):
        n, s, m, c, ns = self.size.get(), self.seed2.get(), self.method2.get(), self.wave2_circular.get(), self.sources2.get()
        self.field2 = self._generate(m, s, n, c, ns)
        self.status_label.config(text=f"Generated Field 2: {m}")
        self._update_plot()

    def combine_fields_ui(self):
        if self.field1 is None or self.field2 is None:
            self.status_label.config(text="Generate both fields first.")
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
        export_obj(self.hybrid, path)
        self.status_label.config(text=f"Exported OBJ: {path}")

    def _update_plot(self):
        self.fig.clf()
        cols, titles = [], []

        if self.field1 is not None:
            cols.append(self.field1)
            titles.append(f"Field 1 – {self.method1.get()}")
        if self.field2 is not None:
            cols.append(self.field2)
            titles.append(f"Field 2 – {self.method2.get()}")
        if self.hybrid is not None:
            cols.append(self.hybrid)
            titles.append(f"Result (Mode: {self.hybrid_mode.get()})")

        for i, f in enumerate(cols):
            ax = self.fig.add_subplot(1, len(cols), i + 1)
            ax.imshow(f, cmap="terrain", origin="lower")
            ax.axis("off")
            ax.set_title(titles[i], color="white", fontsize=10)

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    TerrainUI(root)
    root.mainloop()
