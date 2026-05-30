import sys
import os
sys.path.append(r"d:\Documents\DEEP LEARNING\PROJECT UAS\my-banana-classifier")
from app import load_mobilenet_v3, load_convnext_tiny

m, err_m = load_mobilenet_v3()
c, err_c = load_convnext_tiny()

print("Mob error:", err_m)
print("Conv error:", err_c)
