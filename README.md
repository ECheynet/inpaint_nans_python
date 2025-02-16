## Inpaint NaNs (Python Port)
This repository contains a Python adaptation of John D’Errico’s original MATLAB function, **`inpaint_nans`**, for filling (inpainting) missing or invalid data in arrays. 
The code is useful for image restoration, surface reconstruction, or any application where NaN values need to be replaced in a smooth, physically motivated way.

Original Matlab repository: https://se.mathworks.com/matlabcentral/fileexchange/4551-inpaint_nans

### Contents
- **`inpaint_nans.py`**: The core function, ported from MATLAB to Python.  
- **`test_inpaint_nans.py`**: Example script demonstrating usage.

### Requirements
- Python 3+
- NumPy
- SciPy
- Matplotlib (for plotting)

### Acknowledgments
- Original MATLAB function by **John D’Errico**.

- John D'Errico (2025). inpaint_nans (https://www.mathworks.com/matlabcentral/fileexchange/4551-inpaint_nans), MATLAB Central File Exchange. Retrieved February 16, 2025. 
