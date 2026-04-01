import h5py
import numpy as np
import os

def create_sample_file(filename="data/sample.h5"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with h5py.File(filename, "w") as f:
        # 1. Basic Scalars
        g_scalars = f.create_group("scalars")
        g_scalars.create_dataset("int64", data=np.int64(42))
        g_scalars.create_dataset("float64", data=np.float64(3.141592653589793))
        g_scalars.create_dataset("string_utf8", data=np.array("Hello 🦀".encode("utf-8"), dtype="S")) # Fixed encoding
        dt_u = h5py.string_dtype(encoding='utf-8')
        g_scalars.create_dataset("string_vlen", data="Hello Vlen", dtype=dt_u)

        # 2. 1D Arrays
        g_1d = f.create_group("arrays_1d")
        g_1d.create_dataset("int32", data=np.arange(10, dtype='int32'))
        g_1d.create_dataset("float32", data=np.random.rand(10).astype('float32'))
        
        # 3. N-D Arrays
        g_nd = f.create_group("arrays_nd")
        g_nd.create_dataset("2d_int", data=np.arange(12).reshape(3, 4))
        g_nd.create_dataset("3d_float", data=np.random.rand(2, 3, 4))

        # 4. Compound Types
        g_compound = f.create_group("compound")
        dt_comp = np.dtype([("x", np.int32), ("y", np.float64)])
        data_comp = np.array([(1, 1.5), (2, 2.5)], dtype=dt_comp)
        g_compound.create_dataset("particles", data=data_comp)

        # 5. Enums
        g_enum = f.create_group("enums")
        dt_enum = h5py.enum_dtype({'RED': 0, 'GREEN': 1, 'BLUE': 2}, basetype=np.uint8)
        data_enum = np.array([0, 1, 2, 1, 0], dtype=np.uint8) # Written as raw ints, h5py attaches enum dtype
        dset_enum = g_enum.create_dataset("colors", data=data_enum, dtype=dt_enum)
        g_enum.create_dataset("color_scalar", data=np.uint8(1), dtype=dt_enum)

        # 6. Attributes
        f.attrs['file_owner'] = 'Maintenance Bot'
        g_scalars.attrs['description'] = 'Basic scalar types'
        g_1d['int32'].attrs['units'] = 'meters'
        g_1d['int32'].attrs['scale'] = 1.0

        # 7. Links
        f['link_to_scalars'] = g_scalars
        # Soft links and external links are harder to make portable/robust in quick scripts but let's try soft
        f['soft_link_to_int'] = h5py.SoftLink('/scalars/int64')
        f['soft_link_to_group'] = h5py.SoftLink('/scalars')

        # 8. Custom Types for Rust Introspection Test
        g_custom = f.create_group("custom_types")
        
        # Big Endian
        g_custom.create_dataset("int32_be", data=np.arange(5, dtype='>i4'))
        
        # Custom 6-byte Integer (using low-level h5t)
        try:
            from h5py import h5t, h5s, h5d
            
            # Create a custom 6-byte integer type
            tid = h5t.STD_I32LE.copy()
            tid.set_size(6)
            
            # Create a simple dataspace
            sid = h5s.create_simple((5,))
            
            # Create dataset using low-level API (attached to the group ID)
            h5d.create(g_custom.id, b"int48", tid, sid)
        except Exception as e:
            print(f"Warning: Could not create custom 6-byte integer: {e}")

        # deeply nested group
        g_deep = f.create_group("deeply_nested")
        current = g_deep
        for i in range(10):
            current = current.create_group(f"level_{i}")
        current.create_dataset("final_data", data=np.array([1, 2, 3]))

    print(f"Created {filename}")

if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "data/sample.h5"
    create_sample_file(filename)
