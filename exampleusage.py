import numpy as np
import matplotlib.pyplot as plt
from terrain_algorithms import *

N = 256
GRID_RANGE = 10.0
grid = np.linspace(-GRID_RANGE, GRID_RANGE, N)

# Perlin noise
print("Generating Perlin...")
perlin = perlin2_factory(seed=42)
perlin_field = np.zeros((N, N))
for i, x in enumerate(grid):
    for j, y in enumerate(grid):
        perlin_field[j, i] = fbm_perlin(perlin, x * 0.1, y * 0.1, octaves=5)
perlin_field = (perlin_field - perlin_field.min()) / (perlin_field.max() - perlin_field.min())

# Simplex
print("Generating Simplex...")
simplex = simplex2_factory(seed=42)
simplex_field = np.zeros((N, N))
for i, x in enumerate(grid):
    for j, y in enumerate(grid):
        simplex_field[j, i] = fbm_perlin(simplex, x * 0.1, y * 0.1, octaves=5)
simplex_field = (simplex_field - simplex_field.min()) / (simplex_field.max() - simplex_field.min())

# Diamond-square
print("Generating Diamond–Square...")
ds_field = diamond_square(8, scale=1.0, roughness=0.55, seed=42)
ds_field = (ds_field - ds_field.min()) / (ds_field.max() - ds_field.min())

# Gaussian Process
print("Generating Spectral GP...")
gp_field = spectral_gp_2d(shape=(N, N), length_scale=10.0, variance=1.0, seed=42)

# Multiscale
print("Generating Multiscale...")
multi_field = multiscale_upsampled_field(nx=N, ny=N, levels=6, H=0.7, seed=42)

# Wave interference
print("Generating Wave Interference...")
sources = [(-3, 0, 1.0, 0.0), (3, 0, 1.0, np.pi/2)]
wave_field = interference_grid(sources, grid, grid, k=2*np.pi/4.0, damping=0.02)
rand_wave_field = random_interference_grid(n_sources=8, grid_range=GRID_RANGE, grid_size=N, seed=42)

# Combine fields
print("Combining terrains...")
hybrid_blend = combine_fields(perlin_field, rand_wave_field, method='blend', weight1=0.6, weight2=0.4)
hybrid_mult = combine_fields(perlin_field, rand_wave_field, method='multiply')
hybrid_ridge = combine_fields(perlin_field, rand_wave_field, method='ridge')

# Plot results
fields = [
    ("Perlin", perlin_field),
    ("Simplex", simplex_field),
    ("Diamond–Square", ds_field),
    ("Gaussian Process", gp_field),
    ("Multiscale", multi_field),
    ("Wave Interference", wave_field),
    ("Random Wave Interference", rand_wave_field),
    ("Hybrid Blend", hybrid_blend),
    ("Hybrid Ridge", hybrid_ridge),
]

fig, axs = plt.subplots(3, 3, figsize=(13, 12))
axs = axs.flatten()

for ax, (title, data) in zip(axs, fields):
    im = ax.imshow(data, cmap='terrain', origin='lower')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    fig.colorbar(im, ax=ax, shrink=0.7)

for ax in axs[len(fields):]:
    ax.axis('off')

plt.suptitle("Procedural Terrain Generation Algorithms", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()


# Export terrain as .OBJ for Blender
print("Exporting terrain to OBJ...")
save_heightmap_as_obj(hybrid_blend, "hybrid_blend.obj", scale=(1.0, 1.0, 25.0))
