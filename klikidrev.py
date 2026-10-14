import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import rasterio
import os

# Function to close the application when the Escape key is pressed
def close(event):
    sys.exit()

# Class to represent a Tree object with ID, coordinates (x, y), and height of tree (height)
class Tree:
    def __init__(self, id, x, y, height):
        self.id = id
        self.x = x
        self.y = y
        self.height = height

    # Method to calculate the distance to another tree, weighted by height
    def distance_to(self, another: "Tree", height_weight=1):
        distance = np.sqrt(np.square(self.x - another.x) + np.square(self.y - another.y) + np.square(self.height * height_weight - another.height * height_weight))
        return distance

    # String representation of the Tree object
    def __str__(self):
        return f"{self.id}, height: {self.height}"

# Main application class for matching data
class MatchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data align")  # Set the window title
        self.data2 = None  # Lidar data
        self.modified_data2 = None  # Modified lidar data after transformations
        self.new_trees = pd.DataFrame()#pd.DataFrame(columns=["id", "x", "y", "height"])  # DataFrame to store new trees

        # Create a notebook (tabbed interface)
        self.tab_control = ttk.Notebook(self.root)
        
        # Tab 1: File Input
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text="Upload data")  # Load data tab
        self.create_tab1()
        
        # Tab 2: Scatter Plot, without ref data one button can be assigned to save the changes
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab2, text="Add trees")  # Add trees
        self.create_tab2()
    
        self.tab_control.pack(expand=1, fill="both")  # Pack the notebook into the root window

    # Tab 1: Input Files
    def create_tab1(self):
        label = ttk.Label(self.tab1, text="Select a folder or individual files:")  # Updated label
        label.pack(pady=10)

        #self.file1_path = tk.StringVar()  # Variable to store the path of the first file
        self.file2_path = tk.StringVar()  # Variable to store the path of the second file
        self.geotiff_file_path = tk.StringVar()  # Variable to store the path of the GeoTIFF file
        self.folder_path = tk.StringVar()  # Variable to store the selected folder path

        ttk.Entry(self.tab1, textvariable=self.file2_path, width=50).pack(padx=10, pady=5)  # Entry for ALS tree path
        ttk.Entry(self.tab1, textvariable=self.geotiff_file_path, width=50).pack(padx=10, pady=5)  # Entry for GeoTIFF file path
        ttk.Entry(self.tab1, textvariable=self.folder_path, width=50).pack(padx=10, pady=5)  # Entry for folder path

        file2_button = ttk.Button(self.tab1, text="Lidar data (CSV)", command=self.browse_file2)  # Button to browse for ALS trees
        file2_button.pack(pady=5)

        geotiff_button = ttk.Button(self.tab1, text="Lidar data (GeoTIFF)", command=self.browse_file3)  # Button to browse for GeoTIFF file
        geotiff_button.pack(pady=5)

        folder_button = ttk.Button(self.tab1, text="Select Folder", command=self.browse_folder)  # Button to browse for folder
        folder_button.pack(pady=5)

        load_button = ttk.Button(self.tab1, text="Upload data", command=self.load_files)  # Button to load the files
        load_button.pack(pady=20)

    # Method to browse and select the ALS trees file
    def browse_file2(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            self.file2_path.set(filepath)

    # Method to browse and select the GeoTIFF file
    def browse_file3(self):
        filepath = filedialog.askopenfilename(filetypes=[("GeoTIFF Files", "*.tif;*.tiff")])
        if filepath:
            self.geotiff_file_path.set(filepath)

    # Method to browse and select a folder
    def browse_folder(self):
        folderpath = filedialog.askdirectory()
        if folderpath:
            self.folder_path.set(folderpath)
            self.auto_select_files(folderpath)

    # Method to automatically select files based on the folder name
    def auto_select_files(self, folderpath):
        import os
        folder_name = os.path.basename(folderpath)
        if folder_name.startswith("pl_") and folder_name.count("_") == 2:
            _, plot_id, year = folder_name.split("_")
            las_file = os.path.join(folderpath, f"{folder_name}_las.csv")
            chm_file = os.path.join(folderpath, f"{folder_name}_chm.tif")

            
            # Check if the files exist and set their paths
            if os.path.exists(las_file):
                self.file2_path.set(las_file)
            else:
                messagebox.showerror("Error", f"Lidar data file not found: {las_file}")

            if os.path.exists(chm_file):
                self.geotiff_file_path.set(chm_file)
            else:
                messagebox.showerror("Error", f"GeoTIFF file not found: {chm_file}")
        else:
            messagebox.showerror("Error", "Invalid folder name format. Expected format: pl_<plot_id>_<year>")
      
    # Method to load the selected files and perform initial transformations
    def load_files(self):
        try:
            # Load the ALS tree file data and check for required columns
            self.data2 = pd.read_csv(self.file2_path.get())
            if "X" in self.data2.columns:
                self.data2 = self.data2.rename(columns={"X": "x", "Y": "y", "HEIGHT": "height", "ID": "id"})
            elif "x" in self.data2.columns:
                pass
            else:
                raise ValueError("LiDAR data must contain columns: X, Y, HEIGHT, and ID (or lowercase equivalents).")
            self.data2["als_treecode"] = "original"


            # Check if the GeoTIFF file path is provided
            geotiff_file_path = self.geotiff_file_path.get()
            if geotiff_file_path:  # If a GeoTIFF file is provided, load it
                with rasterio.open(geotiff_file_path) as src:
                    self.geotiff_data = src.read(1)  # Read the first band of the GeoTIFF file
                    self.geotiff_transform = src.transform  # Get the affine transform
                    self.geotiff_data = self.geotiff_data.astype(float)

                # Replace extreme "no data" values with NaN
                self.geotiff_data[self.geotiff_data < -1e+20] = np.nan

            else:
                self.geotiff_data = None  # No GeoTIFF file provided

            # Initialize `self.modified_data1` and `self.modified_data2` with the original data
            self.modified_data2 = self.data2.copy()  # Ensure modified_data2 is initialized

            # Upon upload of the files also reset the 'new_trees' dataframe
            self.new_trees = pd.DataFrame()

            messagebox.showinfo("Success", "Files loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load files: {e}")

    # Tab 2: Scatter Plot
    def create_tab2(self):
        self.fig, self.ax = plt.subplots()  # Create a matplotlib figure and axis
        self.fig.set_size_inches(7, 7)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab2)  # Embed the plot in the Tkinter tab
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # Bind click event to the plot
        self.canvas.mpl_connect("button_press_event", self.add_tree_on_click)

        
        # Slider for CHM maximum height
        self.height_slider = tk.Scale(self.tab2, from_=0, to=50, orient=tk.HORIZONTAL, label="Set CHM display height (m)", command=self.update_plot)
        self.height_slider.set(30)
        self.height_slider.pack(fill=tk.X)
        
        # Bind key press event to modify the slider value
        self.canvas.mpl_connect("key_press_event", self.move_slider_key)

        # Button to plot the data
        plot_button = ttk.Button(self.tab2, text="Show data", command=lambda: self.plot_data(self.ax, self.canvas))
        plot_button.pack(pady=10)

        save_button = ttk.Button(self.tab2, text="Save data", command=lambda: self.save_new_trees())
        save_button.pack(pady=10)

    # Handle clicks on the GeoTIFF plot and add (or remove) a tree
    def add_tree_on_click(self, event):
        if event.inaxes != self.ax or self.geotiff_data is None:
            messagebox.showerror("Error", "Click inside the plot area to add/remove a tree.")
            return

        x, y = event.xdata, event.ydata

        # Left-click adds a tree
        if event.button == 1:
            col, row = ~self.geotiff_transform * (x, y)
            col, row = int(col), int(row)

            if 0 <= row < self.geotiff_data.shape[0] and 0 <= col < self.geotiff_data.shape[1]:
                height = self.geotiff_data[row, col]
                if not np.isnan(height):
                    new_tree_id = self.modified_data2["id"].max() + 1 if not self.modified_data2.empty else 1
                    new_tree = {"id": [new_tree_id], "x": [x], "y": [y], "height": [height], "als_treecode" : ["new"]}
                    self.modified_data2 = pd.concat(
                        [self.modified_data2, pd.DataFrame.from_dict(new_tree)],
                        ignore_index=True
                    )

                    if self.new_trees.empty:
                        self.new_trees = pd.DataFrame.from_dict(new_tree)
                    else:
                        self.new_trees = pd.concat(
                            [self.new_trees, pd.DataFrame.from_dict(new_tree)],
                            ignore_index=True
                        )

                    self.plot_data(self.ax, self.canvas)
                    messagebox.showinfo("Tree Added", f"Tree added at ({x:.2f}, {y:.2f}) with height {height:.2f} m.")
                else:
                    messagebox.showerror("Error", "No height data available at the clicked location.")
            else:
                messagebox.showerror("Error", "Clicked location is outside the GeoTIFF bounds.")

        # Right-click removes the nearest tree
        elif event.button == 3:
            if not self.modified_data2.empty:
                # Compute distances to all trees
                distances = np.sqrt((self.modified_data2["x"] - x) ** 2 + (self.modified_data2["y"] - y) ** 2)
                nearest_idx = distances.idxmin()

                removed_tree = self.modified_data2.loc[nearest_idx]
                self.modified_data2 = self.modified_data2.drop(nearest_idx).reset_index(drop=True)

                # Also remove from new_trees if present
                if not self.new_trees.empty and removed_tree["id"] in self.new_trees["id"].values:
                    self.new_trees = self.new_trees[self.new_trees["id"] != removed_tree["id"]].reset_index(drop=True)

                self.plot_data(self.ax, self.canvas)
                messagebox.showinfo("Tree Removed", f"Removed tree at ({removed_tree['x']:.2f}, {removed_tree['y']:.2f}).")
            else:
                messagebox.showerror("Error", "No trees to remove.")

    # Move the slider by one value left and right upon key presses
    def move_slider_key(self, event):
        if self.geotiff_data is None:
            messagebox.showerror("Error", "Click inside the plot area to add/remove a tree.")
            return
        current_val = self.height_slider.get()

        if event.key == "left":
            # Move slider one step left
            new_val = max(self.height_slider.cget("from"), current_val - 1)
            self.height_slider.set(new_val)

        elif event.key == "right":
            # Move slider one step right
            new_val = min(self.height_slider.cget("to"), current_val + 1)
            self.height_slider.set(new_val)
        
    # Method to save the new trees to a separate CSV file
    def save_new_trees(self):
        if not self.modified_data2.empty:
            # Automatically save the new trees DataFrame in the selected folder
            folder_path = self.folder_path.get()
            if folder_path:
                folder_name = os.path.basename(folder_path)
                new_trees_file = os.path.join(folder_path, f"{folder_name}_las_modified.csv")
                self.modified_data2.to_csv(new_trees_file, sep=",", index=False)
                png_file = os.path.join(folder_path, f"{folder_name}_plot.png")
                self.ax.figure.savefig(png_file)


                messagebox.showinfo("Success", f"New trees saved successfully as {new_trees_file}!")
            else:
                messagebox.showerror("Error", "Folder path not set. Please select a folder first.")
        else:
            messagebox.showerror("Error", "No new trees to save.")

    # Method to plot the data on the scatter plot
    def plot_data(self, ax, canvas):
        if self.data2 is not None:
            ax.clear()

            # Plot GeoTIFF data if available
            if self.geotiff_data is not None:
                #print(f"Plotting GeoTIFF data with min: {np.nanmin(self.geotiff_data)}, max: {np.nanmax(self.geotiff_data)}")

                # Mask NaN values for plotting
                masked_geotiff = np.ma.masked_invalid(self.geotiff_data)
                
                # Set the upper bounds for the masked geotiff to the top recorded height of a recognized las or field tree
                masked_geotiff[masked_geotiff > self.height_slider.get()] = self.height_slider.get()                

                # Plot the GeoTIFF data with the viridis colormap
                ax.imshow(masked_geotiff, cmap="viridis", extent=[
                    self.geotiff_transform[2],
                    self.geotiff_transform[2] + self.geotiff_transform[0] * self.geotiff_data.shape[1],
                    self.geotiff_transform[5] + self.geotiff_transform[4] * self.geotiff_data.shape[0],
                    self.geotiff_transform[5]
                ], alpha=0.5 ) # Reduced alpha for less visibility

           
            # Plot LiDAR data (including added trees)
            ax.scatter(self.modified_data2['x'], self.modified_data2['y'], label="LiDAR data", color='red', s=self.modified_data2["height"]**1.5/4, alpha=0.7)
            if self.new_trees.empty != True:
                ax.scatter(self.new_trees['x'], self.new_trees['y'], label="added trees", color='green', s=self.new_trees["height"]**1.5/4, alpha=0.7)

            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')

            # Set a fixed aspect ratio without resizing the window
            ax.set_aspect("equal", adjustable="datalim")
            ax.autoscale_view()  # Ensure the view is adjusted to fit the data

            ax.legend(bbox_to_anchor=(0.3, 1.15))
            folder_path = self.folder_path.get()
            folder_name = folder_path.split(r"/")[-1]
            ax.set_title(folder_name)
            canvas.draw()
        else:
            messagebox.showerror("Error", "Files not loaded. Please upload the required files first.")

    # Method to update the plot based on slider values
    def update_plot(self, val=None):
        if self.data2 is not None:
            # Re-plot the data
            self.plot_data(self.ax, self.canvas)
        else:
            messagebox.showerror("Error", "Field data not loaded.")  # Show error if field data is not loaded

# Running the application
if __name__ == "__main__":
    root = tk.Tk()
    root.bind('<Escape>', close)  # Bind the Escape key to close the application
    app = MatchingApp(root)
    root.mainloop()

