import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle, Ellipse, Wedge, Polygon
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Create extended weather data with more features
dates = pd.date_range('2025-01-01', '2025-06-01', freq='D')
np.random.seed(42)

# Generate more comprehensive weather data
temperature = np.linspace(25, 47, len(dates)) + np.random.normal(0, 2, len(dates))
humidity = np.linspace(60, 50, len(dates)) + np.random.normal(0, 5, len(dates))
rainfall = np.maximum(0, 2 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + 3 + np.random.normal(0, 1, len(dates)))
wind_speed = np.abs(np.random.normal(15, 5, len(dates)))
pressure = np.random.normal(1015, 5, len(dates))

weather_data = pd.DataFrame({
    'Date': dates,
    'Temperature': temperature,
    'Humidity': humidity,
    'Rainfall': rainfall,
    'Wind_Speed': wind_speed,
    'Pressure': pressure
})

# Create ADVANCED matplotlib dashboard with unused plot types
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(22, 18))
fig.suptitle('ADVANCED MATPLOTLIB VISUALIZATION DASHBOARD\nUnused Plot Types Demonstration', 
             fontsize=20, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.8, wspace=0.4)

# 1. VIOLIN PLOT - Temperature distribution by month
ax1 = fig.add_subplot(gs[0, 0])
weather_data['Month'] = weather_data['Date'].dt.month
monthly_temp_data = [weather_data[weather_data['Month'] == month]['Temperature'] for month in range(1, 7)]
violin_parts = ax1.violinplot(monthly_temp_data, showmeans=True, showmedians=True)
ax1.set_title('Violin Plot: Temperature by Month', fontweight='bold')
ax1.set_xticks(range(1, 7))
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
ax1.set_ylabel('Temperature (°C)')

# Color the violins
for pc in violin_parts['bodies']:
    pc.set_facecolor('lightcoral')
    pc.set_alpha(0.7)

# 2. HEATMAP - Correlation matrix
ax2 = fig.add_subplot(gs[0, 1])
correlation_matrix = weather_data[['Temperature', 'Humidity', 'Rainfall', 'Wind_Speed', 'Pressure']].corr()
im = ax2.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax2.set_xticks(range(len(correlation_matrix.columns)))
ax2.set_yticks(range(len(correlation_matrix.columns)))
ax2.set_xticklabels(correlation_matrix.columns, rotation=45)
ax2.set_yticklabels(correlation_matrix.columns)
ax2.set_title('Heatmap: Variable Correlations', fontweight='bold')

# Add correlation values as text
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        ax2.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', fontweight='bold')

plt.colorbar(im, ax=ax2, label='Correlation Coefficient')

# 3. CONTOUR PLOT - Temperature vs Humidity density
ax3 = fig.add_subplot(gs[0, 2])
x = weather_data['Temperature']
y = weather_data['Humidity']

# Create grid for contour plot
xi = np.linspace(x.min(), x.max(), 50)
yi = np.linspace(y.min(), y.max(), 50)
xi, yi = np.meshgrid(xi, yi)

# Simple 2D histogram for contour data
from scipy.stats import gaussian_kde
values = np.vstack([x, y])
kernel = gaussian_kde(values)
zi = kernel(np.vstack([xi.flatten(), yi.flatten()]))
zi = zi.reshape(xi.shape)

contour = ax3.contourf(xi, yi, zi, levels=15, cmap='viridis')
ax3.contour(xi, yi, zi, levels=15, colors='black', alpha=0.3, linewidths=0.5)
ax3.set_xlabel('Temperature (°C)')
ax3.set_ylabel('Humidity (%)')
ax3.set_title('Contour Plot: Temp-Humidity Density', fontweight='bold')
plt.colorbar(contour, ax=ax3, label='Density')

# 4. POLAR PLOT - Wind direction and speed (simulated)
ax4 = fig.add_subplot(gs[0, 3], projection='polar')
wind_direction = np.random.uniform(0, 2*np.pi, len(dates))
wind_speed = weather_data['Wind_Speed'].values

scatter_polar = ax4.scatter(wind_direction, wind_speed, c=weather_data['Temperature'], 
                           cmap='hot', s=30, alpha=0.7)
ax4.set_title('Polar Plot: Wind Patterns', fontweight='bold', pad=20)
ax4.set_theta_zero_location('N')
ax4.set_theta_direction(-1)
plt.colorbar(scatter_polar, ax=ax4, label='Temperature (°C)')

# 5. 3D SCATTER PLOT
ax5 = fig.add_subplot(gs[1, 0], projection='3d')
scatter_3d = ax5.scatter(weather_data['Temperature'], 
                        weather_data['Humidity'], 
                        weather_data['Rainfall'],
                        c=weather_data['Wind_Speed'], 
                        cmap='plasma', s=40, alpha=0.7)
ax5.set_xlabel('Temperature (°C)')
ax5.set_ylabel('Humidity (%)')
ax5.set_zlabel('Rainfall (mm)')
ax5.set_title('3D Scatter: Multi-variable Analysis', fontweight='bold')
plt.colorbar(scatter_3d, ax=ax5, label='Wind Speed (km/h)')

# 6. STREAMPLOT - Wind flow pattern (simulated)
ax6 = fig.add_subplot(gs[1, 1])
Y, X = np.mgrid[20:50:20j, 20:50:20j]
U = np.sin(X/5) * np.cos(Y/5)  # Simulated U wind component
V = np.cos(X/5) * np.sin(Y/5)  # Simulated V wind component
speed = np.sqrt(U**2 + V**2)

stream = ax6.streamplot(X, Y, U, V, color=speed, cmap='cool', linewidth=1.5, density=2)
ax6.set_xlabel('X Position')
ax6.set_ylabel('Y Position')
ax6.set_title('Streamplot: Wind Flow Patterns', fontweight='bold')
plt.colorbar(stream.lines, ax=ax6, label='Flow Speed')

# 7. ERRORBAR PLOT - Monthly averages with std deviation
ax7 = fig.add_subplot(gs[1, 2])
monthly_avg_temp = weather_data.groupby('Month')['Temperature'].mean()
monthly_std_temp = weather_data.groupby('Month')['Temperature'].std()

months = range(1, 7)
ax7.errorbar(months, monthly_avg_temp, yerr=monthly_std_temp, 
            fmt='o-', color='red', linewidth=2, markersize=8, 
            capsize=5, capthick=2, alpha=0.8)
ax7.set_xticks(months)
ax7.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
ax7.set_xlabel('Month')
ax7.set_ylabel('Temperature (°C)')
ax7.set_title('Errorbar: Monthly Temp ± Std Dev', fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. QUIVER PLOT - Wind vectors
ax8 = fig.add_subplot(gs[1, 3])
x_quiver = np.linspace(0, 10, 8)
y_quiver = np.linspace(0, 10, 8)
X_q, Y_q = np.meshgrid(x_quiver, y_quiver)
U_q = np.sin(X_q) * np.cos(Y_q)  # X component
V_q = np.cos(X_q) * np.sin(Y_q)  # Y component

quiver = ax8.quiver(X_q, Y_q, U_q, V_q, np.sqrt(U_q**2 + V_q**2), 
                   cmap='viridis', scale=20, width=0.005)
ax8.set_xlabel('X Direction')
ax8.set_ylabel('Y Direction')
ax8.set_title('Quiver Plot: Wind Vector Field', fontweight='bold')
plt.colorbar(quiver, ax=ax8, label='Wind Speed')

# 9. FILLED AREA with multiple regions
ax9 = fig.add_subplot(gs[2, 0])
x_fill = np.arange(len(weather_data))
y_upper = weather_data['Temperature'] + weather_data['Temperature'].std()
y_lower = weather_data['Temperature'] - weather_data['Temperature'].std()

ax9.fill_between(x_fill, y_upper, y_lower, alpha=0.3, color='red', label='±1 Std Dev')
ax9.plot(x_fill, weather_data['Temperature'], color='red', linewidth=2, label='Temperature')
ax9.set_xlabel('Days')
ax9.set_ylabel('Temperature (°C)')
ax9.set_title('Filled Area: Temperature Range', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 10. STAIR PLOT - Cumulative rainfall
ax10 = fig.add_subplot(gs[2, 1])
cumulative_rain = weather_data['Rainfall'].cumsum()
ax10.stairs(cumulative_rain, fill=True, color='blue', alpha=0.7)
ax10.set_xlabel('Days')
ax10.set_ylabel('Cumulative Rainfall (mm)')
ax10.set_title('Stair Plot: Cumulative Rainfall', fontweight='bold')
ax10.grid(True, alpha=0.3)

# 11. EVENTPLOT - Rain events
ax11 = fig.add_subplot(gs[2, 2])
rain_events = weather_data[weather_data['Rainfall'] > 0].index.tolist()
ax11.eventplot(rain_events, orientation='horizontal', colors='blue', linewidths=2)
ax11.set_xlabel('Days')
ax11.set_title('Eventplot: Rainy Days Timeline', fontweight='bold')
ax11.set_yticks([0])
ax11.set_yticklabels(['Rain Events'])

# 12. HEXBIN - Temperature vs Humidity density
ax12 = fig.add_subplot(gs[2, 3])
hexbin = ax12.hexbin(weather_data['Temperature'], weather_data['Humidity'], 
                     gridsize=20, cmap='YlOrRd', alpha=0.8)
ax12.set_xlabel('Temperature (°C)')
ax12.set_ylabel('Humidity (%)')
ax12.set_title('Hexbin: Temp-Humidity Density', fontweight='bold')
plt.colorbar(hexbin, ax=ax12, label='Count')

# 13. STEM PLOT - Significant rainfall events (CORRECTED)
ax13 = fig.add_subplot(gs[3, 0])
significant_rain = weather_data[weather_data['Rainfall'] > weather_data['Rainfall'].mean()]

# CORRECTED STEM PLOT - remove use_line_collection parameter
markerline, stemlines, baseline = ax13.stem(significant_rain.index, 
                                           significant_rain['Rainfall'],
                                           basefmt=' ')

plt.setp(stemlines, color='blue', linewidth=2)
plt.setp(markerline, color='darkblue', markersize=6)
ax13.set_xlabel('Days')
ax13.set_ylabel('Rainfall (mm)')
ax13.set_title('Stem Plot: Significant Rainfall Events', fontweight='bold')
ax13.grid(True, alpha=0.3)

# 14. BARBS - Wind barb plot (meteorological standard)
ax14 = fig.add_subplot(gs[3, 1])
x_barb = np.linspace(0, 10, 5)
y_barb = np.linspace(0, 10, 5)
X_b, Y_b = np.meshgrid(x_barb, y_barb)
U_b = np.random.uniform(-5, 5, (5, 5))  # U wind component
V_b = np.random.uniform(-5, 5, (5, 5))  # V wind component

ax14.barbs(X_b, Y_b, U_b, V_b, length=6, barbcolor='blue', flagcolor='red')
ax14.set_xlabel('X Direction')
ax14.set_ylabel('Y Direction')
ax14.set_title('Barbs Plot: Wind Direction/Speed', fontweight='bold')

# 15. PCOLORMESH - 2D data representation
ax15 = fig.add_subplot(gs[3, 2])
# Create 2D temperature-humidity matrix
temp_bins = np.linspace(weather_data['Temperature'].min(), weather_data['Temperature'].max(), 10)
hum_bins = np.linspace(weather_data['Humidity'].min(), weather_data['Humidity'].max(), 10)
H, _, _ = np.histogram2d(weather_data['Temperature'], weather_data['Humidity'], 
                        bins=[temp_bins, hum_bins])

mesh = ax15.pcolormesh(temp_bins, hum_bins, H.T, cmap='hot', shading='auto')
ax15.set_xlabel('Temperature (°C)')
ax15.set_ylabel('Humidity (%)')
ax15.set_title('Pcolormesh: 2D Frequency', fontweight='bold')
plt.colorbar(mesh, ax=ax15, label='Frequency')

# 16. CUSTOM SHAPES and ARTIST
ax16 = fig.add_subplot(gs[3, 3])
ax16.axis('equal')
ax16.set_xlim(0, 10)
ax16.set_ylim(0, 10)

# Add various shapes
circle = Circle((2, 8), 1, color='red', alpha=0.7, label='Hot')
rect = Rectangle((4, 7), 2, 1, color='blue', alpha=0.7, label='Cold')
ellipse = Ellipse((7, 8), 3, 1.5, color='green', alpha=0.7, label='Normal')
polygon = Polygon([[1, 2], [2, 4], [4, 3], [3, 1]], color='orange', alpha=0.7, label='Variable')

ax16.add_patch(circle)
ax16.add_patch(rect)
ax16.add_patch(ellipse)
ax16.add_patch(polygon)

ax16.set_title('Custom Shapes: Weather Patterns', fontweight='bold')
ax16.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("ADVANCED VISUALIZATION TYPES USED IN THIS DASHBOARD:")
print("="*60)
advanced_plots = [
    "1. Violin Plot", "2. Heatmap", "3. Contour Plot", "4. Polar Plot",
    "5. 3D Scatter Plot", "6. Streamplot", "7. Errorbar Plot", 
    "8. Quiver Plot", "9. Filled Area", "10. Stair Plot",
    "11. Eventplot", "12. Hexbin Plot", "13. Stem Plot",
    "14. Barbs Plot", "15. Pcolormesh", "16. Custom Shapes/Artists"
]

for plot in advanced_plots:
    print(f" {plot}")

print(f"\nTotal Visualization Types: {len(advanced_plots)}")
print("This covers nearly all major matplotlib plot types!")