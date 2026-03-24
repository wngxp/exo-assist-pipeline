"""
Convert STL mesh files to Simbody-compatible VTP (ASCII) format.
Writes minimal VTP that Simbody's parser can actually read.

Usage:
    pip install numpy-stl
    python3 convert_stl_to_vtp_simbody.py
"""

import os
import sys
import struct

def read_binary_stl(filepath):
    """Read a binary STL file, return vertices and triangles."""
    with open(filepath, 'rb') as f:
        header = f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]

        vertices = []
        triangles = []
        vertex_map = {}

        for i in range(num_triangles):
            # normal (3 floats) + 3 vertices (9 floats) + attribute (1 short)
            data = struct.unpack('<12fH', f.read(50))
            tri_indices = []
            for v in range(3):
                vx = data[3 + v*3]
                vy = data[4 + v*3]
                vz = data[5 + v*3]
                key = (round(vx, 8), round(vy, 8), round(vz, 8))
                if key not in vertex_map:
                    vertex_map[key] = len(vertices)
                    vertices.append(key)
                tri_indices.append(vertex_map[key])
            triangles.append(tri_indices)

    return vertices, triangles

def write_simbody_vtp(vtp_path, vertices, triangles):
    """Write a VTP file in the minimal ASCII format Simbody expects."""
    num_points = len(vertices)
    num_polys = len(triangles)

    with open(vtp_path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{num_points}" NumberOfPolys="{num_polys}">\n')

        # Points
        f.write('      <Points>\n')
        f.write(f'        <DataArray type="Float32" NumberOfComponents="3" format="ascii">')
        for v in vertices:
            f.write(f' {v[0]} {v[1]} {v[2]}')
        f.write('</DataArray>\n')
        f.write('      </Points>\n')

        # Polys
        f.write('      <Polys>\n')

        # connectivity
        f.write(f'        <DataArray type="Int32" Name="connectivity" format="ascii">')
        for tri in triangles:
            f.write(f' {tri[0]} {tri[1]} {tri[2]}')
        f.write('</DataArray>\n')

        # offsets
        f.write(f'        <DataArray type="Int32" Name="offsets" format="ascii">')
        for i in range(num_polys):
            f.write(f' {(i+1)*3}')
        f.write('</DataArray>\n')

        f.write('      </Polys>\n')
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')

if __name__ == "__main__":
    stl_files = ["base_link.STL", "Link1.STL", "Link2.STL"]

    print("STL -> Simbody-compatible VTP Converter\n")

    for stl_file in stl_files:
        if not os.path.exists(stl_file):
            print(f"  WARNING: {stl_file} not found, skipping.")
            continue

        vtp_file = os.path.splitext(stl_file)[0] + ".vtp"
        vertices, triangles = read_binary_stl(stl_file)
        write_simbody_vtp(vtp_file, vertices, triangles)
        print(f"  {stl_file} -> {vtp_file} ({len(vertices)} vertices, {len(triangles)} triangles)")

    print("\nDone! Move .vtp files into the Geometry/ folder.")
