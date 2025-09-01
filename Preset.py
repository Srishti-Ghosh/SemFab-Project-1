Preset

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk # For better-looking widgets

# --- Import Matplotlib libraries for 3D plotting ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def update_model(*args):
    """
    Reads all input fields and sliders and redraws the 3D model.
    """
    try:
        # Fetch conceptual values from text entries
        source_doping = float(entry_source.get())
        drain_doping = float(entry_drain.get())
        substrate_doping = float(entry_substrate.get())

        # Fetch dimension values from sliders
        fin_height = slider_fin_height.get()
        fin_width = slider_fin_width.get()
        gate_width = slider_gate_width.get()

        # Call the new drawing function with all parameters
        draw_tri_gate_transistor(fin_height, fin_width, gate_width)

    except ValueError:
        messagebox.showerror("Invalid Input", "Please ensure all doping values are valid numbers.")
        return

def draw_tri_gate_transistor(fin_height, fin_width, gate_width):
    """
    Draws a 3D model of a Tri-Gate (FinFET) transistor based on slider values.
    """
    # Clear the previous 3D plot
    ax.cla()

    # --- 1. Define Dimensions and Positions for the 3D Blocks ---
    substrate_dx, substrate_dy, substrate_dz = 10, -4, 10
    oxide_dx, oxide_dy, oxide_dz = 10, -2.5, 10
    fin_dx, fin_dy, fin_dz = fin_width, fin_height, 8
    gate_dx, gate_dy, gate_dz = gate_width, fin_height * 0.5 + 1, 9 # Gate height scales with fin

    sub_x, sub_y, sub_z = 0, 0, 0
    oxide_x, oxide_y, oxide_z = 0, substrate_dy, 0
    fin_x, fin_y, fin_z = (substrate_dx - fin_dx) / 2, substrate_dy, (substrate_dz - fin_dz) / 2
    gate_x, gate_y, gate_z = (substrate_dx - gate_dx) / 2, oxide_dy, (substrate_dz - gate_dz) / 2

    # --- 2. Draw the 3D Blocks for each Region ---
    ax.bar3d(sub_x, sub_y, sub_z, substrate_dx, substrate_dy, substrate_dz,
             color='grey', edgecolor='black', alpha=0.8, shade=True)

    oxide_cutout_width = fin_dx + 0.4
    oxide1_dx = (oxide_dx - oxide_cutout_width) / 2
    oxide2_x = oxide1_dx + oxide_cutout_width
    ax.bar3d(oxide_x, oxide_y, oxide_z, oxide1_dx, oxide_dy, oxide_dz,
             color='lightblue', edgecolor='black', alpha=0.6, shade=True)
    ax.bar3d(oxide2_x, oxide_y, oxide_z, oxide1_dx, oxide_dy, oxide_dz,
             color='lightblue', edgecolor='black', alpha=0.6, shade=True)

    ax.bar3d(fin_x, fin_y, fin_z, fin_dx, fin_dy, fin_dz,
             color='gold', edgecolor='black', alpha=1.0, shade=True)

    ax.bar3d(gate_x, gate_y, gate_z, gate_dx, gate_dy, gate_dz,
             color='lightgreen', edgecolor='black', alpha=0.9, shade=True)

    # --- 3. Add Annotations in 3D Space ---
    ax.text(sub_x + substrate_dx / 2, sub_y + substrate_dy / 2, sub_z - 1,
            "Silicon\nSubstrate", ha='center', va='center', color='white', fontsize=9)
    ax.text(oxide_x + oxide1_dx / 2, oxide_y + oxide_dy / 2, oxide_z + oxide_dz,
            "Oxide", ha='center', va='center', color='black', fontsize=10)
    ax.text(gate_x + gate_dx / 2, gate_y + gate_dy + 0.5, gate_z + gate_dz / 2,
            "Gate", ha='center', va='center', color='black', fontsize=10, zorder=15)
    ax.text(fin_x + fin_dx / 2, fin_y + fin_dy / 2, fin_z - 0.5,
            "Source", ha='center', va='center', color='black', fontsize=10, zorder=15)
    ax.text(fin_x + fin_dx / 2, fin_y + fin_dy / 2, fin_z + fin_dz + 0.5,
            "Drain", ha='center', va='center', color='black', fontsize=10, zorder=15)

    # --- 4. Set Plot Properties ---
    ax.set_box_aspect([1.2, 0.8, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("3D Tri-Gate Transistor (FinFET)", fontsize=14)
    ax.view_init(elev=20, azim=-65)
    canvas.draw()

def draw_initial_view():
    ax.cla(); ax.text2D(0.5, 0.5, "Enter parameters and adjust sliders\nto draw the 3D transistor model.", transform=ax.transAxes, ha="center", va="center", fontsize=10, style='italic', color='gray')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False); canvas.draw()

# --- Main Tkinter Window Setup ---
root = tk.Tk()
root.title("Interactive 3D Tri-Gate Transistor Visualizer")
root.geometry("700x750")

# --- Top Frame for Inputs and Controls ---
top_frame = tk.Frame(root)
top_frame.pack(pady=10, padx=10, fill="x")

# --- Input Frame ---
input_frame = ttk.LabelFrame(top_frame, text="Transistor Properties")
input_frame.pack(side="left", padx=5, pady=5, fill="y")
tk.Label(input_frame, text="Source Doping (cm⁻³):").grid(row=1, column=0, sticky="e", padx=5, pady=2)
entry_source = tk.Entry(input_frame); entry_source.grid(row=1, column=1, padx=5, pady=2)
tk.Label(input_frame, text="Drain Doping (cm⁻³):").grid(row=2, column=0, sticky="e", padx=5, pady=2)
entry_drain = tk.Entry(input_frame); entry_drain.grid(row=2, column=1, padx=5, pady=2)
tk.Label(input_frame, text="Substrate Doping (cm⁻³):").grid(row=3, column=0, sticky="e", padx=5, pady=2)
entry_substrate = tk.Entry(input_frame); entry_substrate.grid(row=3, column=1, padx=5, pady=2)
submit_btn = tk.Button(input_frame, text="Draw Model", command=update_model, bg="lightblue", font=("Arial", 10))
submit_btn.grid(row=4, column=0, columnspan=2, pady=10)

# --- Control Frame (for sliders) ---
control_frame = ttk.LabelFrame(top_frame, text="Model Geometry")
control_frame.pack(side="left", padx=10, pady=5, fill="both", expand=True)
tk.Label(control_frame, text="Fin Height:").grid(row=0, column=0, sticky='w', padx=5)
slider_fin_height = tk.Scale(control_frame, from_=1, to=10, resolution=0.2, orient="horizontal", command=update_model); slider_fin_height.set(4); slider_fin_height.grid(row=0, column=1, sticky='ew')
tk.Label(control_frame, text="Fin Width:").grid(row=1, column=0, sticky='w', padx=5)
slider_fin_width = tk.Scale(control_frame, from_=0.5, to=5, resolution=0.1, orient="horizontal", command=update_model); slider_fin_width.set(1); slider_fin_width.grid(row=1, column=1, sticky='ew')
tk.Label(control_frame, text="Gate Width:").grid(row=2, column=0, sticky='w', padx=5)
slider_gate_width = tk.Scale(control_frame, from_=1, to=8, resolution=0.1, orient="horizontal", command=update_model); slider_gate_width.set(3); slider_gate_width.grid(row=2, column=1, sticky='ew')
control_frame.columnconfigure(1, weight=1) # Make slider column expandable

# --- 3D Plot Setup ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Draw the initial placeholder message
draw_initial_view()

# Start the Tkinter event loop
root.mainloop()