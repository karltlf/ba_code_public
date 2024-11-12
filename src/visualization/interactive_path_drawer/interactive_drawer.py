import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
import pandas as pd

class InteractiveLineDrawer:
    def __init__(self, xlim, ylim, background_lines: pd.DataFrame, num_points=100, window_size=(10, 6)):
        # Create figure and set the window size
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(*window_size)  # Set window size in inches (width, height)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.background_lines = background_lines
        self.num_points = num_points
        self.clicked_points = []  # Store clicked points for the connect feature
        self.dots = []  # Store references to plotted dots for clearing
        self.lines = []  # Store references to lines for preserving them
        self.interpolated_points = []  # To store all interpolated points for each line

        # Plot background lines
        for _, row in self.background_lines.iterrows():
            self.ax.plot(row['x'], row['y'], linestyle='--', color='gray', alpha=0.1)

        # Set up event listeners for mouse clicks
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)

        # Create button to finish the line
        finish_ax = plt.axes([0.1, 0.01, 0.2, 0.05])  # Bottom left
        self.finish_button = Button(finish_ax, 'Finish Line', color='lightgray', hovercolor='green')
        self.finish_button.on_clicked(self.finish_line)

        # Create button to clear points
        clear_ax = plt.axes([0.35, 0.01, 0.2, 0.05])  # Next to finish button
        self.clear_button = Button(clear_ax, 'Clear Points', color='lightgray', hovercolor='yellow')
        self.clear_button.on_clicked(self.clear_points)

        # Create close button
        close_ax = plt.axes([0.85, 0.01, 0.1, 0.05])  # Bottom right
        self.close_button = Button(close_ax, 'Close', color='lightgray', hovercolor='red')
        self.close_button.on_clicked(self.close_plot)

        plt.show()

    def on_press(self, event):
        """Handle mouse button press events for clicking points."""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click
            # Clicking points mode: add a red dot at each click
            point, = self.ax.plot(event.xdata, event.ydata, 'ro')  # Plot red dot for each click
            self.clicked_points.append((event.xdata, event.ydata))
            self.dots.append(point)  # Save the dot for future clearing
            self.fig.canvas.draw()

    def finish_line(self, event):
        """Finish the line by connecting the clicked points."""
        if len(self.clicked_points) < 2:
            print("Not enough points to connect.")
            return

        # Interpolate the points to return the desired number of points
        clicked_points_array = np.array(self.clicked_points)
        distances = np.cumsum(np.sqrt(np.sum(np.diff(clicked_points_array, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)  # Include the starting point
        interpolation_func = interp1d(distances, clicked_points_array, axis=0)

        # Generate equally spaced distances along the line
        new_distances = np.linspace(0, distances[-1], self.num_points)
        interpolated_line = interpolation_func(new_distances)  # Save the interpolated points

        # Store the interpolated points for this line
        self.interpolated_points.append(interpolated_line)

        # Plot the interpolated line
        x_coords, y_coords = interpolated_line[:, 0], interpolated_line[:, 1]
        line, = self.ax.plot(x_coords, y_coords, 'g-', label="Interpolated Line")
        self.ax.legend()
        self.lines.append(line)  # Save the line reference so it is not cleared
        self.fig.canvas.draw()

        # Reset clicked points after finishing the line
        self.clicked_points.clear()
        self.dots.clear()  # Reset dots after line is finished

    def clear_points(self, event):
        """Clear all currently drawn red dots but keep the lines intact."""
        # Remove only the red dots (not the lines)
        for dot in self.dots:
            dot.remove()

        # Clear the list of dots but leave lines intact
        self.dots.clear()
        self.clicked_points.clear()

        # Redraw the canvas after clearing
        self.fig.canvas.draw()

    def close_plot(self, event):
        """Close the plot window without killing the kernel."""
        plt.close(self.fig)

    def get_interpolated_points(self):
        """Return the interpolated points as a list of lists of (x, y) tuples after all lines are finished."""
        if not self.interpolated_points:
            print("No points have been interpolated yet. Please finish drawing the line.")
            return None

        # Return a list of lists of (x, y) tuples for all the lines
        return [list(zip(line[:, 0], line[:, 1])) for line in self.interpolated_points]




def launch_interactive_drawer(xlim, ylim, background_lines: pd.DataFrame, num_points=100, window_size=(10, 6)):
    """
    Launch the interactive line drawer tool in clicking points mode.
    
    Args:
    xlim (tuple): X-axis limits (min, max).
    ylim (tuple): Y-axis limits (min, max).
    background_lines (pd.DataFrame): Background lines as a DataFrame with 'x' and 'y' columns.
    num_points (int): Number of points for interpolating the line.
    window_size (tuple): Width and height of the window in inches (default: (10, 6)).
    
    Returns:
    InteractiveLineDrawer object: The interactive line drawer object to retrieve interpolated points.
    """
    drawer = InteractiveLineDrawer(xlim, ylim, background_lines, num_points, window_size)
    return drawer
