#!/usr/bin/env bash

source omero-env/bin/activate

echo "=== Nuclear option: Install very specific compatible versions ==="

# Remove both numpy and scikit-image
pip uninstall numpy scikit-image -y

# Install a very specific numpy version that definitely works with scikit-image 0.19.3
pip install numpy==1.21.6

# Now install scikit-image 0.19.3
pip install --no-cache-dir scikit-image==0.19.3

# Test
python -c "
import numpy as np
import skimage
print(f'NumPy: {np.__version__}')
print(f'scikit-image: {skimage.__version__}')

# Test the problematic line directly
try:
    import numpy as np
    bool8_exists = hasattr(np, 'bool8')
    print(f'numpy.bool8 exists: {bool8_exists}')
    if bool8_exists:
        print('✅ numpy.bool8 is available')
    else:
        print('❌ numpy.bool8 is missing')
except Exception as e:
    print(f'Error checking bool8: {e}')

# Test skimage import
from skimage import morphology
from lavlab.omero.images import get_large_recon, load_image_smart
print('🎉 Success!')
"