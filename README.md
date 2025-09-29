# 2D Mask Layout Editor

  

Welcome to the 2D Mask Layout Editor\! This is a powerful and intuitive desktop application for designing complex 2D layouts for microfabrication, PCB design, and other engineering tasks. Built with Python and PyQt5, it bridges the gap between abstract 2D designs and their 3D physical form.

The application saves projects in a human-readable JSON format and supports importing and exporting industry-standard GDSII and OASIS file formats.

*A quick look at the user interface and drawing workflow.*

-----

## ✨ Key Features

  * 🎨 **Comprehensive Drawing Tools:** Create rectangles, complex polygons, circles, and paths with snap-to-grid precision.
  * 🏗️ **Hierarchical Design:** Organize your layout into reusable **Cells** and instance them throughout your project to build complex designs efficiently.
  * ✂️ **Advanced Shape Manipulation:**
      * Perform **Boolean** operations (Union, Subtract, Intersect).
      * Fine-tune polygons with an interactive **Vertex Editor**.
      * Automatically round corners with the **Fillet** tool.
  * 📚 **Full Layer Management:** Control layer visibility, color, and drawing order with ease.
  * 🔄 **GDSII & OASIS Support:** Import existing layouts and export your designs for fabrication or use in other tools (requires `gdstk`).
  * 🧊 **Interactive 3D Preview:** Instantly extrude your 2D layout into a 3D view to visualize layer thicknesses and how they stack up.
  * 🖱️ **Intuitive UI:** A modern, ribbon-style interface with dockable panels for an efficient and customizable workflow.

-----

## 🚀 Getting Started

Get up and running in just a few steps.

### Prerequisites

  * Python 3.6+
  * The following libraries: `PyQt5`, `gdstk`, `matplotlib`

### Installation

1.  **Clone or Download:** Get a copy of the project files.
2.  **Install Dependencies:** Open your terminal and run the single installation command:
    ```bash
    pip install PyQt5 gdstk matplotlib
    ```
3.  **Setup Icons:** Create an `icons/` folder in the same directory as the script and populate it with your favorite `.svg` icon set (e.g., from [Feather Icons](https://feathericons.com/)). This makes the UI look great\!
4.  **Run the Application:**
    ```bash
    python layout_editor.py
    ```

-----

## 🗺️ A Quick Tour of the UI

The interface is designed to be clean and efficient. Here are the main components:

| Component | Description |
|---|---|
| **Tool Ribbon** | At the top, find all your tools and actions organized into tabs like `Main`, `Draw`, and `View`. |
| **Canvas** | Your main workspace. **Middle-click to pan** and **scroll to zoom**. |
| **Cells Panel (Left)** | Manage your design hierarchy. **Double-click** to edit a cell, or **drag-and-drop** onto the canvas to create an instance. |
| **Layers Panel (Right)**| Control all your layers. Toggle visibility, change colors, and set the active drawing layer. |

**Pro Tip:** **Double-click any shape** on the canvas to open a dialog for precise numerical editing\!

-----

## 📂 File Structure

A simple file structure is all that's needed to get started.

```text
/your-project-folder
|-- layout_editor.py        # The main application script
|-- /icons/                 # Folder for UI icons
|   |-- save.svg
|   |-- folder.svg
|   |-- ... (and other .svg icons)
```
