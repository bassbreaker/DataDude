import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator, Rbf

FILENAME = 'test1.csv'
SENSOR_MIN = 0
SENSOR_MAX = 110
SENSOR_RES = 5
TEMPERATURE_MIN = 15
TEMPERATURE_MAX = 35
TEMPERATURE_RES = 0.25

my_data = np.genfromtxt(FILENAME, delimiter=',')
sensor_data = my_data[1:,0]
temperature_data = my_data[1:,1]
power_data = my_data[1:,2]

sensor_arr = np.linspace(SENSOR_MIN, SENSOR_MAX, int((SENSOR_MAX-SENSOR_MIN)/SENSOR_RES+1))
temperature_arr = np.linspace(TEMPERATURE_MIN, TEMPERATURE_MAX, int((TEMPERATURE_MAX-TEMPERATURE_MIN)/TEMPERATURE_RES+1))
sensor_grid, temperature_grid = np.meshgrid(sensor_arr, temperature_arr, indexing='ij')

grid_table = griddata((sensor_data, temperature_data), power_data, (sensor_grid, temperature_grid), method='linear')
grid_interp = RegularGridInterpolator((sensor_grid[:,0], temperature_grid[0,:]), grid_table, bounds_error=False)

rbf3 = Rbf(sensor_data, temperature_data, power_data, function="linear", smooth=2)
rbf_table = rbf3(sensor_grid, temperature_grid)
rbf_interp = RegularGridInterpolator((sensor_grid[:,0], temperature_grid[0,:]), rbf_table, method='linear')

# calculate RMSError
grid_error = 0
grid_out_bound = 0
rbf_error = 0
rbf_out_bound = 0
for i in range(1, len(sensor_data)):
    check = grid_interp([sensor_data[i],temperature_data[i]])[0]
    if not np.isnan(check):
        grid_error += (check - power_data[i])**2
    else:
        grid_out_bound += 1
    check = rbf_interp([sensor_data[i],temperature_data[i]])[0]
    if not np.isnan(check):
        rbf_error += (check - power_data[i])**2
    else:
        rbf_out_bound += 1

grid_error = (grid_error/(len(my_data)-1-grid_out_bound))**0.5
rbf_error = (rbf_error/(len(my_data)-1-rbf_out_bound))**0.5

print(f"sensres: {SENSOR_RES}, tempres: {TEMPERATURE_RES}")
print(f"grid: error: {grid_error}, outbound: {grid_out_bound}")
print(f"rbf: error: {rbf_error}, outbound: {rbf_out_bound}")

# plot it out
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(sensor_data, temperature_data, power_data, c='tab:orange')
ax.plot_wireframe(sensor_grid, temperature_grid, rbf_table, alpha=0.5)
plt.show()