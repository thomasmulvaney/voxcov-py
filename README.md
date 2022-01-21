# voxcov-py

Python bindings for voxcov.

## Quick start

Clone `voxcov` first.

Run `maturin develop --release` to build and install the plugin.  Use the `--release` flag to get an optimized build.


## Coverage example
Import the library

```python
import voxcov as vc
```

Create a map.

```python
apix = [1.0, 1.0, 1.0]
origin = [0.0, 0.0, 0.0]

# The maximum axis length
size = 256
vox_map = vc.Map(apix, origin, size)
```

Add and delete spheres.

```python
# Add a sphere of radius 3.5 at (10,10,10)
vox_map.add_sphere([10.0, 10.0, 10.0], 3.5, lambda vox: print("Adding A Covered: ", vox))

# Add another sphere which overlaps mostly of radius 3.5 at (11,10,10)
vox_map.add_sphere([11.0, 10.0, 10.0], 3.5, lambda vox: print("Adding B Covered: ", vox))

# Delete sphere which overlaps mostly of radius 3.5 at (11,10,10)
vox_map.del_sphere([11.0, 10.0, 10.0], 3.5, lambda vox: print("Deleting B uncovered: ", vox))
```


## Blur example
Import the library

```python
import voxcov as vc
```

Create a blurrer.  Note, the API is a bit clunky.  Named parameters would be nice.

```python
blurred_map = vc.BlurMap(
    [1.084,1.084,1.084], # The cell dimensions
    [0,0,0],             # The origin
    [256,256,256],       # The size, 256^3
    5.1,                 # Sigma for gaussian
                         # e.g. (sigma_coef * resolution)
    4.0                  # The number of sigma to cutoff at
)
```

Stick atoms in.

```python
# We pass the center and scale of each gaussian.
blurred_map.add_gaussian([3,4,5], 10)
blurred_map.add_gaussian([4,5,3], 12)
```

Get a numpy array out.

```python
blurred_map.to_numpy()
```
