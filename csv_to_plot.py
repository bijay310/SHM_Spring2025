import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "jordainaexperimentoutside_trial3_real_life_coordinates.csv"  
df = pd.read_csv(file_path)

# Create plots
plt.figure(figsize=(12, 6))

# Plot X coordinate over time
plt.subplot(2, 1, 1)
plt.plot(df['time/s'], df['x0_real/mm'], label='X Position', color='blue')
plt.title('X Real Coordinate Over Time')
plt.xlabel('Time (s)')
plt.ylabel('X Position (mm)')
plt.grid(True)

# Plot Y coordinate over time
plt.subplot(2, 1, 2)
plt.plot(df['time/s'], df['y0_real/mm'], label='Y Position', color='green')
plt.title('Y Real Coordinate Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Y Position (mm)')
plt.grid(True)

#display
plt.tight_layout()
plt.show()
