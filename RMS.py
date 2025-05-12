import pandas as pd
import numpy as np
from tkinter import Tk, filedialog

root = Tk()
root.withdraw()  
file_path = filedialog.askopenfilename(
    title="Select CSV File",
    filetypes=[("CSV files", "*.csv")]
)

# Load the CSV file
df = pd.read_csv(file_path)

# Reference value in mm 
# ( this is true for the 3 story structure, and it needs to be changed 
# based on the eyeballed/ observed value of the displacement in mm )
reference_value = 19.05

# Calculate RMS Error
rms_error = np.sqrt(np.mean((df['x0_real/mm'] - reference_value) ** 2))

# Calculate RMS Error Percentage
percent_rms_error = (rms_error / reference_value) * 100

# Output the results
print(f"RMS Error: {rms_error:.4f} mm")
print(f"RMS Error %: {percent_rms_error:.2f}%")
