# terrain_viewer_trimesh.py
import trimesh
import sys

def view_obj(filepath):
    mesh = trimesh.load(filepath)
    scene = mesh.scene()
    scene.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python terrain_viewer_trimesh.py <terrain.obj>")
        sys.exit(1)
    view_obj(sys.argv[1])
