# Units
MM = 1e-3
UM = 1e-6
NM = 1e-9

# Lens parameters
GRID_N = 255 # grid pixel size 
LENS_D = 40 * UM
PIX_SIZE = 350 * NM
WAVL = 532 * NM
EFL = 100 * UM
Z = 100 * UM
HFOV = 7 # degrees

# Field sampling
H_OBJ = 32
W_OBJ = 32
FIELD_STRATEGY = "block"   # "block" or "full"
BLOCK_SIZE = 4             # only used if strategy == "block"

# Image construction
PSF_WINDOW_N = 63
TUKEY_ALPHA = 0.05

