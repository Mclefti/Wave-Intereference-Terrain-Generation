# Wave Intereference Terrain Generation

📦 Requirements

Python 3.9+
Install dependencies via pip:

pip install numpy matplotlib noise pywavelets tk


(Optional: if you use Blender integration or 3D OBJ export, these are already supported via NumPy output.)

🚀 How to Run

Clone or download this repository, then in your terminal:

python main.py


The graphical UI will launch automatically.



🧭 Interface Overview
Section	Description
Field 1 (Base)	The first terrain field (e.g., Perlin, Simplex, etc.)
Field 2 (Modifier)	The second terrain field used to blend or modify Field 1
Wave Sources	Controls how many wave origins (emitters) are used in Wave Interference mode
Circular Waves	Toggles between plane waves and circular wave propagation
Combination Mode	Determines how Field 1 and Field 2 are mixed — blend, multiply, ridge, hybrid blend, etc.
OBJ Export	Saves the generated hybrid field as a 3D .obj mesh suitable for import into Blender
Grid Size	Sets the terrain resolution (e.g., 128, 256, 512)



🌊 Algorithms Supported
Algorithm	Description
Perlin	Smooth coherent noise ideal for rolling hills
Simplex	Faster, less grid-like variant of Perlin noise
Fractal Subdivision	Midpoint displacement for jagged mountainous terrain
Wave Interference	Superposition of sinusoidal waves — mimics interference or dune-like patterns
Hybrid Ridge	Combines fields emphasizing ridge-like features
Hybrid Blend	Weighted average combination for smoother transitions


🧩 Combination Modes
Mode	Effect
blend	Weighted average between both fields
multiply	Multiplies field intensities (increases contrast)
ridge	Emphasizes peaks and ridges
hybrid blend	Combines low and high frequency from both fields


🎨 Color Legend
Color	Meaning
🟩 Green	Lowlands
🟨 Yellow	Midlands
🟫 Brown	Hills
⚪ White	Peaks
🌊 Bright / Dark	Constructive / Destructive interference (for wave fields)



🧠 Notes

For wave interference, default is 8 sources (n_sources=8), each with random amplitude, frequency, and phase.

You can toggle between plane or circular wave origins.

Exported OBJ files can be opened directly in Blender (File → Import → Wavefront (.obj)).

🛠️ File Structure
dual_field_terrain/
│
├── main.py            # Main application with UI and generation logic
├── noise_utils.py     # Optional (if separated noise/generation utilities)
├── requirements.txt   # List of dependencies
└── README.md          # Project documentation (this file)

💡 Tips

Try combining Perlin (Field 1) with WaveInterference (Field 2) using hybrid ridge mode.

Increase grid size for more detail (but slower render).

Randomize seeds for unique landscapes every time.

📜 License

MIT License © 2025 — You’re free to use, modify, and distribute this tool for learning and creative projects.