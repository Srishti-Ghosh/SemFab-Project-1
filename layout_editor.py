import sys
import math
import json
import copy
import uuid
import os
import skimage
from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Dict, List, Tuple, Optional
from uuid import UUID
from skimage import measure
from scipy.ndimage import gaussian_filter

# --- 3D Dependencies ---
try:
    import pyqtgraph.opengl as gl
    from skimage import measure  # <--- Defines 'measure'
    import numpy as np
    _has_3d_deps = True          # <--- Defines '_has_3d_deps'
except ImportError:
    print("Warning: 3D libraries (pyqtgraph, skimage) not found.")
    _has_3d_deps = False
    measure = None
    gl = None
# This is a stub for the gdstk library if it's not installed.
# For full functionality, install it: pip install gdstk
try:
    import gdstk
except ImportError:
    print("Warning: gdstk library not found. GDS/OAS, fillet, boolean features, and some simulation aspects will be disabled.")
    gdstk = None

# Added for the 2_5d view logic and Level Set
try:
    from matplotlib.path import Path
except ImportError:
    print("Warning: matplotlib library not found. 2.5D view and Level Set initialization will be disabled.")
    Path = None

# Added for Level Set Simulation
try:
    from scipy.ndimage import distance_transform_edt
    from skimage.measure import find_contours
    _has_level_set_deps = True
except ImportError:
    print("Warning: numpy, scipy, or scikit-image not found. Level Set simulation features will be disabled. 3D Voxel View disabled.")
    np = None
    distance_transform_edt = None
    find_contours = None
    _has_level_set_deps = False


from PyQt5.QtCore import Qt, QRectF, QPointF, QSize, QSizeF, QMimeData, pyqtSignal, QLineF, QUrl
from PyQt5.QtGui import (QBrush, QPen, QColor, QPolygonF, QCursor, QPainter, QPixmap, QIcon, QPainterPath,
                         QDrag, QPalette, QFont, QTransform, QDesktopServices)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QColorDialog,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPolygonItem,
    QGraphicsItem, QToolBar, QPushButton, QLabel, QLineEdit, QGridLayout,
    QSpinBox, QCheckBox, QDialog, QVBoxLayout, QDialogButtonBox, QToolTip, QProgressBar, QApplication,
    QDockWidget, QListWidget, QListWidgetItem, QWidget, QHBoxLayout, QMessageBox, QDoubleSpinBox, QGraphicsEllipseItem, QComboBox,
    QInputDialog, QTabWidget, QFrame, QToolButton, QGridLayout, QSlider, QGraphicsLineItem, QMenu, QGraphicsSceneMouseEvent, QOpenGLWidget
)
from PyQt5.QtSvg import QSvgGenerator

# -------------------------- Data model --------------------------

@dataclass
class Layer:
    name: str
    color: Tuple[int, int, int]
    visible: bool = True
    fill_pattern: str = "solid"  # <-- ADDED for mask styling
    thickness_2_5d: float = 10.0 # <-- ADDED for 2.5D view

@dataclass(eq=False)
class Poly:
    layer: str
    points: List[Tuple[float, float]]
    name: Optional[str] = field(default=None)
    uuid: UUID = field(default_factory=uuid.uuid4, init=False, repr=False)

@dataclass(eq=False)
class Ellipse:
    layer: str
    rect: Tuple[float, float, float, float]  # x, y, w, h
    name: Optional[str] = field(default=None)
    uuid: UUID = field(default_factory=uuid.uuid4, init=False, repr=False)

@dataclass(eq=False)
class Ref:
    cell: str
    origin: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    magnification: float = 1.0

@dataclass
class Cell:
    polygons: List[Poly] = field(default_factory=list)
    ellipses: List[Ellipse] = field(default_factory=list)
    references: List[Ref] = field(default_factory=list)
    etch_pairs: List[Tuple[str, str, float]] = field(default_factory=list)

@dataclass
class Project:
    units: str = "um"
    layers: List[Layer] = field(default_factory=list)
    cells: Dict[str, Cell] = field(default_factory=dict)
    top: str = "TOP"
    grid_pitch: int = 50
    canvas_width: int = 2000
    canvas_height: int = 1500
    layer_by_name: Dict[str, Layer] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.refresh_layer_map()

    def refresh_layer_map(self):
        self.layer_by_name = {l.name: l for l in self.layers}


# -------------------------- Dialogs --------------------------

class WelcomeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to the Layout Editor")
        self.setMinimumSize(450, 300) # Made it slightly taller
        self.choice = None
        self.open_path = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 30)
        layout.setSpacing(10)

        logo_label = QLabel()
        logo_pixmap = self._create_themed_icon("layout", 64)
        logo_label.setPixmap(logo_pixmap.pixmap(QSize(64, 64)))
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        title = QLabel("2D Mask Layout Editor")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; padding-top: 10px;")
        layout.addWidget(title)

        subtitle = QLabel("Create a new project or open an existing file to begin.")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 10pt; color: gray; padding-bottom: 20px;")
        layout.addWidget(subtitle)

        layout.addStretch()

        is_dark_theme = self.palette().color(QPalette.Window).value() < 128
        hover_color = "#404040" if is_dark_theme else "#F0F0F0"
        border_color = "#555" if is_dark_theme else "#CCC"

        button_stylesheet = f"""
            QPushButton {{
                padding: 12px;
                text-align: left;
                font-size: 13px;
                background-color: transparent;
                border: 1px solid {border_color};
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """

        btn_new = QPushButton(self._create_themed_icon("file-plus", 22), " Create New Project")
        btn_new.setIconSize(QSize(22, 22))
        btn_new.setStyleSheet(button_stylesheet)
        btn_new.clicked.connect(self.new_project)
        layout.addWidget(btn_new)

        btn_open = QPushButton(self._create_themed_icon("folder", 22), " Open Existing File...")
        btn_open.setIconSize(QSize(22, 22))
        btn_open.setStyleSheet(button_stylesheet)
        btn_open.clicked.connect(self.open_file)
        layout.addWidget(btn_open)

        layout.addStretch()

    def _create_themed_icon(self, icon_name, size=32):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
        icon_path = os.path.join(script_dir, 'icons', f'{icon_name}.svg')
        pixmap = QPixmap(icon_path)
        if pixmap.isNull():
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.transparent)
        is_dark = self.palette().color(QPalette.Window).value() < 128
        if is_dark:
            painter = QPainter(pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
            painter.fillRect(pixmap.rect(), Qt.white)
            painter.end()
        return QIcon(pixmap)

    def new_project(self):
        self.choice = 'new'
        self.accept()

    def open_file(self):
        file_filter = "Layout Files (*.json *.gds *.oas);;JSON Project (*.json);;GDSII Files (*.gds);;OASIS Files (*.oas)"
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter)
        if path:
            self.choice = 'open'
            self.open_path = path
            self.accept()

class GridDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project Setup")
        layout = QGridLayout(self)
        layout.addWidget(QLabel("Grid Spacing:"), 0, 0)
        self.spacing_input = QSpinBox()
        self.spacing_input.setRange(1, 1000); self.spacing_input.setValue(50)
        layout.addWidget(self.spacing_input, 0, 1)

        layout.addWidget(QLabel("Canvas Width:"), 1, 0)
        self.size_input_w = QSpinBox()
        self.size_input_w.setRange(500, 50000); self.size_input_w.setValue(2000)
        layout.addWidget(self.size_input_w, 1, 1)

        layout.addWidget(QLabel("Canvas Height:"), 2, 0)
        self.size_input_h = QSpinBox()
        self.size_input_h.setRange(500, 50000); self.size_input_h.setValue(1500)
        layout.addWidget(self.size_input_h, 2, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 3, 0, 1, 2)

    def get_values(self):
        return self.spacing_input.value(), self.size_input_w.value(), self.size_input_h.value()

class ShapeEditDialog(QDialog):
    def __init__(self, kind, params, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit {kind.capitalize()}")
        self.kind = kind; self.params = params
        layout = QVBoxLayout()
        self.input_fields = {}

        if kind == "rect":
            for label in ["x", "y", "w", "h"]:
                fld = QDoubleSpinBox(); fld.setRange(-1e6, 1e6); fld.setValue(params[label]); fld.setDecimals(3)
                layout.addWidget(QLabel(f"{label.upper()}:")); layout.addWidget(fld)
                self.input_fields[label] = fld
        elif kind == "circle":
            for label in ["center_x", "center_y", "w", "h"]:
                fld = QDoubleSpinBox(); fld.setRange(-1e6, 1e6); fld.setValue(params[label]); fld.setDecimals(3)
                layout.addWidget(QLabel(f"{label.replace('_', ' ').title()}:")); layout.addWidget(fld)
                self.input_fields[label] = fld
        elif kind == "poly":
            self.pt_edits = []
            for i, (x, y) in enumerate(params["points"]):
                fldx, fldy = QDoubleSpinBox(), QDoubleSpinBox()
                for fld in [fldx, fldy]: fld.setRange(-1e6, 1e6); fld.setDecimals(3)
                fldx.setValue(x); fldy.setValue(y)
                row = QHBoxLayout()
                row.addWidget(QLabel(f"V{i+1}:")); row.addWidget(fldx); row.addWidget(fldy)
                layout.addLayout(row)
                self.pt_edits.append((fldx, fldy))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons); self.setLayout(layout)

    def get_values(self):
        if self.kind in ["rect", "circle"]: return {k: v.value() for k, v in self.input_fields.items()}
        if self.kind == "poly": return {"points": [(fx.value(), fy.value()) for fx, fy in self.pt_edits]}
        return None

class FilletDialog(QDialog):
    def __init__(self, parent=None, default_radius=10.0):
        super().__init__(parent)
        self.setWindowTitle("Fillet Polygon")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Fillet Radius:"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setDecimals(3) # Allow for more precision
        self.radius_spin.setRange(0.001, 10000.0)

        # Set the spinbox's value to the passed-in default
        self.radius_spin.setValue(default_radius)

        layout.addWidget(self.radius_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_radius(self): return self.radius_spin.value()

# RENAMED from Interactive3DView
class Interactive2_5dView(QGraphicsView):
    def __init__(self, scene, parent_dialog):
        super().__init__(scene)
        self.parent_dialog = parent_dialog
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self._last_pan_point = QPointF()
        #self.setViewport(QOpenGLWidget())
        #self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

    def mousePressEvent(self, event):
        if event.button() in [Qt.LeftButton, Qt.RightButton]:
            self._last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            delta = event.pos() - self._last_pan_point
            self.parent_dialog.angle_y += delta.x() * 0.5
            self.parent_dialog.angle_x += delta.y() * 0.5
            self.parent_dialog.draw_2_5d_view() # RENAMED
        self._last_pan_point = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        self.parent_dialog.zoom_scale *= zoom_factor
        self.parent_dialog.draw_2_5d_view() # RENAMED

# RENAMED from ThreeDViewDialog
class TwoPointFiveDViewDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Interactive 2.5D Preview") # RENAMED
        base_height = 500
        height_per_layer = 30
        num_layers = len(self.project.layers)
        dynamic_height = base_height + (num_layers * height_per_layer)
        self.setGeometry(100, 100, 800, dynamic_height)

        self.angle_x, self.angle_y, self.zoom_scale = 35.0, -45.0, 1.0
        # RENAMED attribute
        self.layer_thicknesses = {layer.name: getattr(layer, 'thickness_2_5d', 10) for layer in self.project.layers}
        self.sliders = {}

        main_layout = QVBoxLayout(self)
        top_panel = QWidget()
        top_panel_layout = QHBoxLayout(top_panel)
        top_panel_layout.setContentsMargins(0, 0, 0, 0)
        sliders_frame = QFrame(); sliders_frame.setFrameShape(QFrame.StyledPanel)
        sliders_layout = QGridLayout(sliders_frame)
        sliders_layout.addWidget(QLabel("<b>Layer Thickness (units)</b>"), 0, 0, 1, 3) # Changed label
        row = 1
        for layer in self.project.layers:
            label, slider = QLabel(layer.name), QSlider(Qt.Horizontal)
            # RENAMED attribute
            seed = getattr(layer, 'thickness_2_5d', self.layer_thicknesses.get(layer.name, 10))
            seed_int = max(1, int(round(seed)))
            slider.setRange(1, max(100, seed_int)); slider.setValue(seed_int)
            slider.valueChanged.connect(self.on_slider_change)
            value_label = QLabel(f"{slider.value()}") # Changed label
            slider.value_label = value_label
            sliders_layout.addWidget(label, row, 0)
            sliders_layout.addWidget(slider, row, 1)
            sliders_layout.addWidget(value_label, row, 2)
            self.sliders[layer.name] = slider; row += 1
        sliders_layout.setRowStretch(row, 1)

        views_frame = QFrame(); views_frame.setFrameShape(QFrame.StyledPanel)
        views_layout = QVBoxLayout(views_frame)
        views_layout.addWidget(QLabel("<b>Standard Views & Actions</b>"))
        btn_grid = QGridLayout()
        btn_iso, btn_top = QPushButton("2.5D / Iso"), QPushButton("Top") # RENAMED
        btn_front, btn_side = QPushButton("Front"), QPushButton("Side")
        btn_iso.clicked.connect(self.set_iso_view); btn_top.clicked.connect(self.set_top_view)
        btn_front.clicked.connect(self.set_front_view); btn_side.clicked.connect(self.set_side_view)
        btn_grid.addWidget(btn_iso, 0, 0); btn_grid.addWidget(btn_top, 0, 1)
        btn_grid.addWidget(btn_front, 1, 0); btn_grid.addWidget(btn_side, 1, 1)
        views_layout.addLayout(btn_grid)
        btn_save = QPushButton("Save Image..."); btn_save.clicked.connect(self.save_image)
        views_layout.addWidget(btn_save); views_layout.addStretch()
        top_panel_layout.addWidget(sliders_frame, stretch=2); top_panel_layout.addWidget(views_frame, stretch=1)
        main_layout.addWidget(top_panel)

        self.scene = QGraphicsScene()
        self.view = Interactive2_5dView(self.scene, self) # RENAMED
        main_layout.addWidget(self.view)

        self.draw_2_5d_view(fit_view=True) # RENAMED

    def set_view_angles(self, x, y):
        self.angle_x, self.angle_y = x, y
        self.zoom_scale = 1.0
        self.draw_2_5d_view(fit_view=True) # RENAMED

    def set_iso_view(self): self.set_view_angles(35.0, -45.0)
    def set_top_view(self): self.set_view_angles(0, 0)
    def set_front_view(self): self.set_view_angles(-90, 0)
    def set_side_view(self): self.set_view_angles(0, -90)

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "2_5d_view.png", "PNG (*.png);;JPG (*.jpg)") # RENAMED default filename
        if path and not self.view.grab().save(path):
            QMessageBox.warning(self, "Save Error", f"Could not save image to:\n{path}")

    def on_slider_change(self):
        for name, slider in self.sliders.items():
            self.layer_thicknesses[name] = slider.value()
            slider.value_label.setText(f"{slider.value()}") # Changed label

            layer = next((l for l in self.project.layers if l.name == name), None)
            if layer:
                layer.thickness_2_5d = float(slider.value()) # RENAMED attribute

        self.draw_2_5d_view() # RENAMED

    def project_point(self, x, y, z):
        view_size = self.view.viewport().size()
        aspect_ratio = 1.0
        if view_size.height() > 0:
            aspect_ratio = view_size.width() / view_size.height()
        rad_y, rad_x = math.radians(self.angle_y), math.radians(self.angle_x)
        cos_y, sin_y, cos_x, sin_x = math.cos(rad_y), math.sin(rad_y), math.cos(rad_x), math.sin(rad_x)
        x1 = x * cos_y + z * sin_y; z1 = -x * sin_y + z * cos_y
        y1 = y * cos_x - z1 * sin_x
        return QPointF(x1 * self.zoom_scale * aspect_ratio, -y1 * self.zoom_scale)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.draw_2_5d_view() # RENAMED

    # RENAMED from draw_3d_view
    def draw_2_5d_view(self, fit_view=False):
        self.scene.clear()

        def get_all_shapes_as_polygons(cell, origin=(0, 0), rotation=0, mag=1.0):
            # This helper function is unchanged
            shapes = []
            t = QTransform().translate(origin[0], origin[1]).rotate(rotation).scale(mag, mag)
            for p_data in cell.polygons:
                if (layer := self.project.layer_by_name.get(p_data.layer)) and layer.visible:
                    poly = t.map(QPolygonF([QPointF(*p) for p in p_data.points]))
                    shapes.append((p_data.layer, [(p.x(), p.y()) for p in poly]))
            for e_data in cell.ellipses:
                if (layer := self.project.layer_by_name.get(e_data.layer)) and layer.visible:
                    rect = QRectF(*e_data.rect)
                    poly = QPolygonF([QPointF(rect.center().x() + rect.width()/2 * math.cos(i/16*math.pi),
                                              rect.center().y() + rect.height()/2 * math.sin(i/16*math.pi)) for i in range(32)])
                    shapes.append((e_data.layer, [(p.x(), p.y()) for p in t.map(poly)]))
            for ref in cell.references:
                if ref.cell in self.project.cells:
                    ref_origin_transformed = t.map(QPointF(*ref.origin))
                    shapes.extend(get_all_shapes_as_polygons(self.project.cells[ref.cell],
                                                              (ref_origin_transformed.x(), ref_origin_transformed.y()),
                                                              ref.rotation + rotation,
                                                              ref.magnification * mag))
            return shapes

        # === PASS 1: Calculate all final 2D geometric primitives ===
        all_shapes = get_all_shapes_as_polygons(self.project.cells[self.project.top])
        if not all_shapes: return

        active_cell_2_5d = self.project.cells[self.project.top]
        geometric_primitives = []
        if gdstk and hasattr(active_cell_2_5d, 'etch_pairs') and active_cell_2_5d.etch_pairs:
            etch_masks = {}
            used_mask_layers = set()
            for mask, sub, depth in active_cell_2_5d.etch_pairs:
                used_mask_layers.add(mask)
                if sub not in etch_masks: etch_masks[sub] = []
                # Ensure points are valid before creating gdstk.Polygon
                valid_polys = [gdstk.Polygon(p) for L, p in all_shapes if L == mask and p]
                etch_masks[sub].extend([(poly, depth) for poly in valid_polys])


            for layer_name, points in all_shapes:
                if layer_name in used_mask_layers: continue
                if not points: continue # Skip empty polygons

                if layer_name in etch_masks and etch_masks[layer_name]:
                    substrate_poly = gdstk.Polygon(points)
                    masks_for_sub = [m[0] for m in etch_masks[layer_name]]
                    depth = etch_masks[layer_name][0][1]

                    un_etched = gdstk.boolean(substrate_poly, masks_for_sub, 'not')
                    cavity_floors = gdstk.boolean(substrate_poly, masks_for_sub, 'and')

                    # Handle cases where boolean might return non-list
                    if isinstance(un_etched, gdstk.Polygon): un_etched = [un_etched]
                    if isinstance(cavity_floors, gdstk.Polygon): cavity_floors = [cavity_floors]

                    for part in un_etched or []: geometric_primitives.append({'layer': layer_name, 'points': part.points.tolist(), 'mod': 0})
                    for floor in cavity_floors or []: geometric_primitives.append({'layer': layer_name, 'points': floor.points.tolist(), 'mod': -depth})
                else:
                    geometric_primitives.append({'layer': layer_name, 'points': points, 'mod': 0})
        else:
            geometric_primitives = [{'layer': l, 'points': p, 'mod': 0} for l, p in all_shapes if p] # Ensure points exist

        if not geometric_primitives: return

        # === PASS 2: Build the complete, final height map from the primitives ===
        height_map = {}
        grid_step = 20.0 # Adjust for accuracy vs performance
        layer_order = {layer.name: i for i, layer in enumerate(self.project.layers)}
        geometric_primitives.sort(key=lambda s: layer_order.get(s['layer'], -1))

        if Path: # Check if Path is available
            for shape in geometric_primitives:
                layer_name, points_2d, thickness_mod = shape['layer'], shape['points'], shape['mod']
                layer = self.project.layer_by_name.get(layer_name)
                if not layer or not points_2d: continue

                full_thickness = self.layer_thicknesses.get(layer_name, 10)
                thickness = max(0, full_thickness + thickness_mod)
                path = Path(points_2d)

                try:
                    p_min_x_g = int(min(p[0] for p in points_2d) // grid_step)
                    p_max_x_g = int(max(p[0] for p in points_2d) // grid_step)
                    p_min_y_g = int(min(p[1] for p in points_2d) // grid_step)
                    p_max_y_g = int(max(p[1] for p in points_2d) // grid_step)
                except ValueError: # Handle empty points list
                    continue

                for gx in range(p_min_x_g, p_max_x_g + 1):
                    for gy in range(p_min_y_g, p_max_y_g + 1):
                        if path.contains_point(((gx + 0.5) * grid_step, (gy + 0.5) * grid_step)):
                            base_z = height_map.get((gx, gy), 0)
                            height_map[(gx, gy)] = base_z + thickness

        all_faces = []
        try:
            all_points_flat = [pt for shape in geometric_primitives for pt in shape['points']]
            if not all_points_flat: return # Nothing to render
            min_x, max_x = min(p[0] for p in all_points_flat), max(p[0] for p in all_points_flat)
            min_y, max_y = min(p[1] for p in all_points_flat), max(p[1] for p in all_points_flat)
        except ValueError: # Handle case where geometric_primitives is empty
            return
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

        render_height_map = {}
        if Path: # Check if Path is available
            for shape in geometric_primitives:
                layer_name, points_2d, thickness_mod = shape['layer'], shape['points'], shape['mod']
                layer = self.project.layer_by_name.get(layer_name)
                if not layer or not layer.visible or not points_2d: continue

                full_thickness = self.layer_thicknesses.get(layer_name, 10)
                thickness = max(0, full_thickness + thickness_mod)
                color = QColor(*layer.color)

                base_z = 0
                path = Path(points_2d)
                try:
                    p_min_x_g = int(min(p[0] for p in points_2d) // grid_step)
                    p_max_x_g = int(max(p[0] for p in points_2d) // grid_step)
                    p_min_y_g = int(min(p[1] for p in points_2d) // grid_step)
                    p_max_y_g = int(max(p[1] for p in points_2d) // grid_step)
                except ValueError: continue

                footprint_z_values = [0]
                for gx in range(p_min_x_g, p_max_x_g + 1):
                    for gy in range(p_min_y_g, p_max_y_g + 1):
                        if path.contains_point(((gx + 0.5) * grid_step, (gy + 0.5) * grid_step)):
                            footprint_z_values.append(render_height_map.get((gx, gy), 0))
                base_z = max(footprint_z_values)

                centered_pts = [(p[0] - center_x, p[1] - center_y) for p in points_2d]
                floor_pts = [(*p, base_z) for p in centered_pts]
                ceiling_pts = [(*p, base_z + thickness) for p in centered_pts]
                all_faces.append((ceiling_pts, color.lighter(110)))
                if base_z > 0: all_faces.append((floor_pts, color.darker(120)))
                for i in range(len(floor_pts)):
                    p1, p2 = floor_pts[i], floor_pts[(i + 1) % len(floor_pts)]
                    p3, p4 = ceiling_pts[(i + 1) % len(floor_pts)], ceiling_pts[i]
                    all_faces.append(([p1, p2, p3, p4], color))

                new_top_z = base_z + thickness
                for gx in range(p_min_x_g, p_max_x_g + 1):
                    for gy in range(p_min_y_g, p_max_y_g + 1):
                        if path.contains_point(((gx + 0.5) * grid_step, (gy + 0.5) * grid_step)):
                            render_height_map[(gx, gy)] = new_top_z

        design_width = max_x - min_x
        design_height = max_y - min_y
        total_h = max(height_map.values()) if height_map else 0
        if total_h > 0:
            v = [(-design_width/2,-design_height/2,0), (design_width/2,-design_height/2,0), (design_width/2,design_height/2,0), (-design_width/2,design_height/2,0),
                 (-design_width/2,-design_height/2,total_h), (design_width/2,-design_height/2,total_h), (design_width/2,design_height/2,total_h), (-design_width/2,design_height/2,total_h)]
            box_faces = [(0,1,2,3), (4,5,6,7), (0,3,7,4), (1,2,6,5), (0,1,5,4), (2,3,7,6)]
            for face in box_faces: all_faces.append(([v[i] for i in face], QColor(150,150,150,40)))

        rad_y, rad_x = math.radians(self.angle_y), math.radians(self.angle_x)
        cos_y, sin_y, cos_x, sin_x = math.cos(rad_y), math.sin(rad_y), math.cos(rad_x), math.sin(rad_x)
        def get_viewspace_z(p):
            x, y, z = p; z1 = -x * sin_y + z * cos_y; return y * sin_x + z1 * cos_x
        all_faces.sort(key=lambda face: sum(get_viewspace_z(p) for p in face[0]) / len(face[0]))

        z_offset = -total_h / 2.0
        for points_2_5d, col in all_faces:
            points_offset = [(p[0], p[1], p[2] + z_offset) for p in points_2_5d]
            points_2d = [self.project_point(*p) for p in points_offset]
            self.scene.addPolygon(QPolygonF(points_2d), QPen(col.darker(110), 0), QBrush(col))

        axis_length = max(design_width, design_height, total_h) * 0.75
        origin_proj = self.project_point(0, 0, z_offset)
        axes = [((axis_length,0,0),"red","X"), ((0,axis_length,0),"green","Y"), ((0,0,axis_length),"blue","Z")]
        for axis, color_str, name in axes:
            end = self.project_point(axis[0], axis[1], axis[2] + z_offset)
            self.scene.addLine(QLineF(origin_proj, end), QPen(QColor(color_str), 0))
            label = self.scene.addText(name); label.setDefaultTextColor(QColor(color_str)); label.setPos(end)
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations)

        if fit_view:
            bounds = self.scene.itemsBoundingRect()
            if bounds.isValid():
                margin = max(10, bounds.width() * 0.05) # Ensure some margin even if width is 0
                self.view.fitInView(bounds.adjusted(-margin, -margin, margin, margin), Qt.KeepAspectRatio)


# -------------------------- Graphics items --------------------------

class SceneItemMixin:
    def __init__(self, *args, **kwargs): pass
    def customMousePressEvent(self, event): pass
    def customMouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsItem.ItemIsMovable:
             # Check if scene and view exist before accessing
            if self.scene() and self.scene().views():
                view = self.scene().views()[0]
                # Compare scene positions
                if event.scenePos() != event.buttonDownScenePos(Qt.LeftButton):
                    view.parent_win.update_data_from_item_move(self)


class RectItem(QGraphicsRectItem, SceneItemMixin):
    def __init__(self, rect, layer, data_obj, selectable=True):
        QGraphicsRectItem.__init__(self, rect)
        SceneItemMixin.__init__(self)
        self.layer, self.base_color, self.data_obj = layer, QColor(*layer.color), data_obj
        flags = QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable if selectable else QGraphicsItem.GraphicsItemFlags(0)
        self.setFlags(flags)
        self.refresh_appearance(selected=False)
        if self.data_obj.name: self.setToolTip(self.data_obj.name)

    def mousePressEvent(self, event): super().mousePressEvent(event); self.customMousePressEvent(event)
    def mouseReleaseEvent(self, event): super().mouseReleaseEvent(event); self.customMouseReleaseEvent(event)

    def refresh_appearance(self, selected=False):
        # UPDATED to handle fill_pattern
        color = self.base_color.lighter(130) if selected else self.base_color
        color.setAlpha(180)
        brush = QBrush(color)
        pattern = getattr(self.layer, 'fill_pattern', 'solid')
        if pattern == 'hatch': brush.setStyle(Qt.BDiagPattern)
        elif pattern == 'cross': brush.setStyle(Qt.CrossPattern)
        elif pattern == 'dots': brush.setStyle(Qt.Dense7Pattern)
        else: brush.setStyle(Qt.SolidPattern)
        self.setBrush(brush)
        self.setPen(QPen(Qt.black, 0))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange: self.refresh_appearance(selected=bool(value))
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if event is None: event = QGraphicsSceneMouseEvent()
        super().mouseDoubleClickEvent(event)
        if self.scene() and self.scene().views(): # Check scene and views exist
            self.scene().views()[0].parent_win.show_properties_dialog_for_item(self)


class PolyItem(QGraphicsPolygonItem, SceneItemMixin):
    def __init__(self, poly, layer, data_obj, selectable=True):
        QGraphicsPolygonItem.__init__(self, poly)
        SceneItemMixin.__init__(self)
        self.layer, self.base_color, self.data_obj = layer, QColor(*layer.color), data_obj
        flags = QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable if selectable else QGraphicsItem.GraphicsItemFlags(0)
        self.setFlags(flags)
        self.refresh_appearance(selected=False)
        if self.data_obj.name: self.setToolTip(self.data_obj.name)

    def mousePressEvent(self, event): super().mousePressEvent(event); self.customMousePressEvent(event)
    def mouseReleaseEvent(self, event): super().mouseReleaseEvent(event); self.customMouseReleaseEvent(event)

    def refresh_appearance(self, selected=False):
        # UPDATED to handle fill_pattern
        color = self.base_color.lighter(130) if selected else self.base_color
        color.setAlpha(180)
        brush = QBrush(color)
        pattern = getattr(self.layer, 'fill_pattern', 'solid')
        if pattern == 'hatch': brush.setStyle(Qt.BDiagPattern)
        elif pattern == 'cross': brush.setStyle(Qt.CrossPattern)
        elif pattern == 'dots': brush.setStyle(Qt.Dense7Pattern)
        else: brush.setStyle(Qt.SolidPattern)
        self.setBrush(brush)
        self.setPen(QPen(Qt.black, 0))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange: self.refresh_appearance(selected=bool(value))
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if event is None: event = QGraphicsSceneMouseEvent()
        super().mouseDoubleClickEvent(event)
        if self.scene() and self.scene().views(): # Check scene and views exist
            self.scene().views()[0].parent_win.show_properties_dialog_for_item(self)

# -------------------------- Process Dialog (Add this before MainWindow) --------------------------

class ProcessDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Process Step")
        self.project = project
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        form_layout = QGridLayout()

        # Step Selection
        self.combo_step = QComboBox()
        self.combo_step.addItems(["Deposition", "Etch", "Doping", "Oxidation", "Lithography"])
        self.combo_step.currentTextChanged.connect(self._update_ui_state)
        
        # Type Selection (Sub-types)
        self.combo_type = QComboBox()
        
        # Layer Selection
        self.combo_layer1 = QComboBox() # Primary Layer (Target or Mask)
        self.combo_layer2 = QComboBox() # Secondary Layer (Substrate)
        self.update_layer_lists()

        # Parameters
        self.lbl_param1 = QLabel("Thickness (µm):")
        self.spin_param1 = QDoubleSpinBox()
        self.spin_param1.setRange(-1000, 1000); self.spin_param1.setValue(0.5)
        
        self.lbl_param2 = QLabel("Depth/Time:")
        self.spin_param2 = QDoubleSpinBox()
        self.spin_param2.setRange(0, 10000); self.spin_param2.setValue(1.0)

        # Layout Assembly
        row = 0
        form_layout.addWidget(QLabel("Process Step:"), row, 0); form_layout.addWidget(self.combo_step, row, 1); row+=1
        form_layout.addWidget(QLabel("Type/Method:"), row, 0); form_layout.addWidget(self.combo_type, row, 1); row+=1
        
        self.lbl_layer1 = QLabel("Target Layer:")
        form_layout.addWidget(self.lbl_layer1, row, 0); form_layout.addWidget(self.combo_layer1, row, 1); row+=1
        
        self.lbl_layer2 = QLabel("Substrate Layer:")
        form_layout.addWidget(self.lbl_layer2, row, 0); form_layout.addWidget(self.combo_layer2, row, 1); row+=1
        
        form_layout.addWidget(self.lbl_param1, row, 0); form_layout.addWidget(self.spin_param1, row, 1); row+=1
        form_layout.addWidget(self.lbl_param2, row, 0); form_layout.addWidget(self.spin_param2, row, 1); row+=1

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_ui_state(self.combo_step.currentText())

    def update_layer_lists(self):
        layer_names = [l.name for l in self.project.layers]
        self.combo_layer1.clear(); self.combo_layer1.addItems(layer_names)
        self.combo_layer2.clear(); self.combo_layer2.addItems(layer_names)

    def _update_ui_state(self, step_name):
        self.combo_type.clear()
        
        # Defaults
        self.lbl_layer2.setVisible(False); self.combo_layer2.setVisible(False)
        self.lbl_param2.setVisible(False); self.spin_param2.setVisible(False)
        self.lbl_param1.setVisible(True); self.spin_param1.setVisible(True)

        if step_name == "Deposition":
            self.combo_type.addItems(["PVD", "CVD", "Spin-on", "Electroplating"])
            self.lbl_layer1.setText("Target Layer:")
            self.lbl_param1.setText("Thickness:")

        elif step_name == "Etch":
            self.combo_type.addItems(["Wet Etch", "RIE (Dry)", "DRIE", "Ion Milling"])
            self.lbl_layer1.setText("Mask Layer:")
            self.lbl_layer2.setVisible(True); self.lbl_layer2.setText("Substrate (to be etched):")
            self.combo_layer2.setVisible(True)
            self.lbl_param1.setVisible(False); self.spin_param1.setVisible(False) # Etch usually defined by depth
            self.lbl_param2.setVisible(True); self.lbl_param2.setText("Etch Depth:")
            self.spin_param2.setVisible(True)

        elif step_name == "Doping":
            self.combo_type.addItems(["Ion Implantation", "Diffusion"])
            self.lbl_layer1.setText("Mask Layer:")
            self.lbl_param1.setText("Dose (cm⁻²):")
            self.spin_param1.setRange(1e10, 1e16); self.spin_param1.setValue(1e13)
            self.lbl_param2.setVisible(True); self.lbl_param2.setText("Energy (keV):")
            self.spin_param2.setVisible(True); self.spin_param2.setValue(50)

        elif step_name == "Oxidation":
            self.combo_type.addItems(["Dry", "Wet"])
            self.lbl_layer1.setText("Substrate:")
            self.lbl_param1.setText("Target Thickness:")

    def get_values(self):
        return {
            "step": self.combo_step.currentText(),
            "type": self.combo_type.currentText(),
            "layer1": self.combo_layer1.currentText(),
            "layer2": self.combo_layer2.currentText() if self.combo_layer2.isVisible() else None,
            "param1": self.spin_param1.value() if self.spin_param1.isVisible() else None,
            "param2": self.spin_param2.value() if self.spin_param2.isVisible() else None
        }

class CircleItem(QGraphicsEllipseItem, SceneItemMixin):
    def __init__(self, rect, layer, data_obj, selectable=True):
        QGraphicsEllipseItem.__init__(self, rect)
        SceneItemMixin.__init__(self)
        self.layer, self.base_color, self.data_obj = layer, QColor(*layer.color), data_obj
        flags = QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable if selectable else QGraphicsItem.GraphicsItemFlags(0)
        self.setFlags(flags)
        self.refresh_appearance(selected=False)
        if self.data_obj.name: self.setToolTip(self.data_obj.name)

    def mousePressEvent(self, event): super().mousePressEvent(event); self.customMousePressEvent(event)
    def mouseReleaseEvent(self, event): super().mouseReleaseEvent(event); self.customMouseReleaseEvent(event)

    def refresh_appearance(self, selected=False):
        # UPDATED to handle fill_pattern
        color = self.base_color.lighter(130) if selected else self.base_color
        color.setAlpha(180)
        brush = QBrush(color)
        pattern = getattr(self.layer, 'fill_pattern', 'solid')
        if pattern == 'hatch': brush.setStyle(Qt.BDiagPattern)
        elif pattern == 'cross': brush.setStyle(Qt.CrossPattern)
        elif pattern == 'dots': brush.setStyle(Qt.Dense7Pattern)
        else: brush.setStyle(Qt.SolidPattern)
        self.setBrush(brush)
        self.setPen(QPen(Qt.black, 0))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange: self.refresh_appearance(selected=bool(value))
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if event is None: event = QGraphicsSceneMouseEvent()
        super().mouseDoubleClickEvent(event)
        if self.scene() and self.scene().views(): # Check scene and views exist
            self.scene().views()[0].parent_win.show_properties_dialog_for_item(self)


class RefItem(QGraphicsItem):
    def __init__(self, ref, cell, project, selectable=True):
        super().__init__()
        self.ref, self.cell, self.project = ref, cell, project
        self._drag_start_pos = QPointF()
        flags = QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable if selectable else QGraphicsItem.GraphicsItemFlags(0)
        self.setFlags(flags)
        self.setPos(QPointF(*self.ref.origin))
        self.setRotation(self.ref.rotation); self.setScale(self.ref.magnification)
        self._draw_children()

    def _draw_children(self):
        # Clear existing children first
        for item in self.childItems():
            item.setParentItem(None)
            # Optionally delete them if they are managed only here
            # item.deleteLater() # Be careful with ownership

        for p_data in self.cell.polygons:
            if (layer := self.project.layer_by_name.get(p_data.layer)) and layer.visible:
                PolyItem(QPolygonF([QPointF(*pt) for pt in p_data.points]), layer, p_data, False).setParentItem(self)
        for e_data in self.cell.ellipses:
            if (layer := self.project.layer_by_name.get(e_data.layer)) and layer.visible:
                CircleItem(QRectF(*e_data.rect), layer, e_data, False).setParentItem(self)
        for r_data in self.cell.references:
            if r_data.cell in self.project.cells:
                # Prevent infinite recursion if a cell references itself directly or indirectly
                # Basic check: avoid direct self-reference rendering inside itself
                if r_data.cell != self.project.cells[self.ref.cell]: # Check if the referenced cell is the same as the parent Ref's cell
                    RefItem(r_data, self.project.cells[r_data.cell], self.project, selectable=False).setParentItem(self)
                else:
                    print(f"Warning: Skipped rendering self-reference of cell '{r_data.cell}' inside itself.")


    def mousePressEvent(self, event): super().mousePressEvent(event); self._drag_start_pos = self.pos()
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsItem.ItemIsMovable and self.pos() != self._drag_start_pos:
             # Check scene and view exist
            if self.scene() and self.scene().views():
                 self.scene().views()[0].parent_win.update_data_from_item_move(self)


    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene() and self.scene().views():
            if self.scene().views()[0].parent_win.chk_snap.isChecked():
                return self.scene().views()[0].snap_point(value)
        return super().itemChange(change, value)

    def boundingRect(self): return self.childrenBoundingRect() or QRectF(-5, -5, 10, 10) # Fallback needed
    def paint(self, painter, option, widget):
        if self.isSelected():
            painter.setPen(QPen(Qt.darkCyan, 0, Qt.DashLine))
            painter.setBrush(Qt.NoBrush); painter.drawRect(self.boundingRect())

# -------------------------- Custom List Widget for Dragging --------------------------

class CellListWidget(QListWidget):
    """Custom list widget to ensure the cell name is packaged as plain text for drag-and-drop."""
    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item:
            mimeData = QMimeData()
            mimeData.setText(item.text()) # Package the cell name as plain text
            drag = QDrag(self)
            drag.setMimeData(mimeData)
            drag.exec_(Qt.CopyAction | Qt.LinkAction) # Start the drag

# -------------------------- View/Scene with Rulers --------------------------

class Ruler(QWidget):
    def __init__(self, orientation, canvas, parent=None):
        super().__init__(parent)
        self.canvas, self.orientation = canvas, orientation
        self.ruler_font = QFont("Arial", 8)
        self.setFixedHeight(30) if orientation == Qt.Horizontal else self.setFixedWidth(30)

    def paintEvent(self, event):
        if not hasattr(self.canvas, 'parent_win') or not self.canvas.parent_win.project: return # Added check
        painter = QPainter(self); painter.setFont(self.ruler_font)
        bg, tick = self.palette().color(QPalette.Window), self.palette().color(QPalette.WindowText)
        painter.fillRect(self.rect(), bg)
        try: # Add try-except for robustness if scene/view not fully ready
            visible = self.canvas.mapToScene(self.canvas.viewport().rect()).boundingRect()
            zoom = self.canvas.transform().m11()
        except Exception:
            return
        pitch = float(self.canvas.parent_win.project.grid_pitch)
        if pitch <= 0: return
        req_pitch = 8
        draw_pitch = pitch # Start with actual grid pitch
        while (draw_pitch * zoom) < req_pitch: draw_pitch *= 5 # Increase drawn pitch if too dense
        while (draw_pitch * zoom) > req_pitch * 5: draw_pitch /= 5 # Decrease drawn pitch if too sparse
        painter.setPen(QPen(tick, 1))

        if self.orientation == Qt.Horizontal:
            start = math.floor(visible.left() / draw_pitch) * draw_pitch
            end = math.ceil(visible.right() / draw_pitch) * draw_pitch
            x = start
            while x <= end:
                vx = self.canvas.mapFromScene(QPointF(x, 0)).x()
                is_major = abs(x % (pitch * 5)) < (draw_pitch * 0.1) # Check if it's a multiple of 5*pitch
                h = 10 if is_major else 5
                painter.drawLine(vx, self.height(), vx, self.height() - h)
                if is_major and zoom * (pitch * 5) > 40: # Only draw text if spaced enough
                    painter.drawText(vx + 2, self.height() - 12, f"{x:.0f}")
                x += draw_pitch

        else: # Vertical
            start = math.floor(visible.top() / draw_pitch) * draw_pitch
            end = math.ceil(visible.bottom() / draw_pitch) * draw_pitch
            y = start
            while y <= end:
                vy = self.canvas.mapFromScene(QPointF(0, y)).y()
                is_major = abs(y % (pitch * 5)) < (draw_pitch * 0.1) # Check if it's a multiple of 5*pitch
                w = 10 if is_major else 5
                painter.drawLine(self.width(), vy, self.width() - w, vy)
                if is_major and zoom * (pitch * 5) > 40: # Only draw text if spaced enough
                    painter.save()
                    painter.translate(self.width() - 12, vy - 2); painter.rotate(90)
                    painter.drawText(0, 0, f"{-y:.0f}") # Display negative Y upwards
                    painter.restore()
                y += draw_pitch

class CanvasContainer(QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.h_ruler, self.v_ruler = Ruler(Qt.Horizontal, self.canvas), Ruler(Qt.Vertical, self.canvas)
        layout = QGridLayout(); layout.setSpacing(0); layout.setContentsMargins(0,0,0,0)
        corner = QWidget(); corner.setFixedSize(30, 30); corner.setStyleSheet("background-color: palette(window);")
        layout.addWidget(corner, 0, 0); layout.addWidget(self.h_ruler, 0, 1)
        layout.addWidget(self.v_ruler, 1, 0); layout.addWidget(self.canvas, 1, 1)
        self.setLayout(layout)
        # Connect signals
        self.canvas.horizontalScrollBar().valueChanged.connect(self.h_ruler.update)
        self.canvas.verticalScrollBar().valueChanged.connect(self.v_ruler.update)
        self.canvas.zoomChanged.connect(self.h_ruler.update)
        self.canvas.zoomChanged.connect(self.v_ruler.update)

class Canvas(QGraphicsView):
    MODES = {"select": 1, "rect": 2, "poly": 3, "circle": 4, "move": 5, "measure": 6, "path": 7}
    zoomChanged = pyqtSignal()

    def __init__(self, scene, get_active_layer, parent=None):
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing); self.setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse); self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        #self.setViewport(QOpenGLWidget())
        self.mode = self.MODES["select"]; self.start_pos = None; self.temp_item = None
        self.temp_poly_points: List[QPointF] = []
        self.temp_path_points: List[QPointF] = []
        self.get_active_layer, self.parent_win = get_active_layer, parent
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._panning = False
        self._pan_start = QPointF()


    def drawBackground(self, painter: QPainter, rect: QRectF):
        super().drawBackground(painter, rect)
        if not self.parent_win.project: return
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        zoom = self.transform().m11()
        bg, minor_c, major_c = self.palette().color(self.backgroundRole()), self.palette().color(self.foregroundRole()), self.palette().color(self.foregroundRole())
        minor_c.setAlphaF(0.2); major_c.setAlphaF(0.4)
        painter.fillRect(rect, bg)
        pitch = float(self.parent_win.project.grid_pitch)
        if pitch <= 0: return
        req_pitch = 8 # Min pixels between grid lines
        draw_pitch = pitch
        while (draw_pitch * zoom) < req_pitch: draw_pitch *= 5
        while (draw_pitch * zoom) > req_pitch * 5: draw_pitch /= 5
        minor_pen, major_pen = QPen(minor_c, 0), QPen(major_c, 0)
        left, right = visible.left(), visible.right()
        top, bottom = visible.top(), visible.bottom()

        x_start = math.floor(left / draw_pitch) * draw_pitch
        x_end = math.ceil(right / draw_pitch) * draw_pitch
        x = x_start
        while x <= x_end:
            is_major = abs(x % (pitch * 5)) < (draw_pitch * 0.1) if pitch > 0 else (x==0)
            painter.setPen(major_pen if is_major else minor_pen)
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            x += draw_pitch

        y_start = math.floor(top / draw_pitch) * draw_pitch
        y_end = math.ceil(bottom / draw_pitch) * draw_pitch
        y = y_start
        while y <= y_end:
            is_major = abs(y % (pitch * 5)) < (draw_pitch * 0.1) if pitch > 0 else (y==0)
            painter.setPen(major_pen if is_major else minor_pen)
            painter.drawLine(QPointF(left, y), QPointF(right, y))
            y += draw_pitch

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor); self.zoomChanged.emit()

    def set_mode(self, mode_name: str):
        self.mode = self.MODES.get(mode_name, self.MODES["select"])
        is_move = self.mode == self.MODES["move"]
        self.setDragMode(QGraphicsView.RubberBandDrag if self.mode == self.MODES["select"] else QGraphicsView.NoDrag)
        for item in self.scene().items():
            if item.flags() & QGraphicsItem.ItemIsSelectable:
                item.setFlag(QGraphicsItem.ItemIsMovable, is_move)
        self.temp_cancel()

    def temp_cancel(self):
        if self.temp_item: self.scene().removeItem(self.temp_item)
        self.temp_item, self.start_pos = None, None
        self.temp_poly_points.clear()
        self.temp_path_points.clear()
        self.viewport().setCursor(Qt.ArrowCursor)

    def snap_point(self, p: QPointF) -> QPointF:
        if not self.parent_win.chk_snap.isChecked(): return p
        pitch = self.parent_win.project.grid_pitch
        if pitch <= 0: return p
        return QPointF(round(p.x() / pitch) * pitch, round(p.y() / pitch) * pitch)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning, self._pan_start = True, event.pos(); self.setCursor(Qt.ClosedHandCursor)
            event.accept(); return
        scene_pos = self.snap_point(self.mapToScene(event.pos()))
        if event.button() == Qt.LeftButton:
            if self.mode in [self.MODES["rect"], self.MODES["circle"]]:
                self.start_pos = scene_pos
                Klass = QGraphicsRectItem if self.mode == self.MODES["rect"] else QGraphicsEllipseItem
                self.temp_item = Klass(QRectF(self.start_pos, self.start_pos))
                self.temp_item.setPen(QPen(Qt.gray, 0, Qt.DashLine)); self.scene().addItem(self.temp_item)
            elif self.mode == self.MODES["measure"]:
                self.start_pos = scene_pos
                self.temp_item = QGraphicsLineItem(QLineF(self.start_pos, self.start_pos))
                self.temp_item.setPen(QPen(QColor(255,100,0), 2, Qt.DashLine)); self.scene().addItem(self.temp_item)
            elif self.mode == self.MODES["poly"]:
                self.temp_poly_points.append(scene_pos)
                if self.temp_item: self.scene().removeItem(self.temp_item)
                if len(self.temp_poly_points) > 1:
                    poly = QPolygonF(self.temp_poly_points)
                    self.temp_item = self.scene().addPolygon(poly, QPen(Qt.gray, 0, Qt.DashLine))
                elif self.temp_poly_points: # Show first point marker
                    marker_size = max(2, self.parent_win.project.grid_pitch / 10 * self.transform().m11()) # Scale marker size slightly
                    marker_rect = QRectF(scene_pos - QPointF(marker_size/2, marker_size/2), QSizeF(marker_size, marker_size))
                    self.temp_item = self.scene().addRect(marker_rect, QPen(Qt.gray, 0), QBrush(Qt.gray))

            elif self.mode == self.MODES["path"]:
                self.temp_path_points.append(scene_pos)
                if self.temp_item: self.scene().removeItem(self.temp_item)
                if len(self.temp_path_points) > 1:
                    poly = QPolygonF(self.temp_path_points)
                    path = QPainterPath(); path.moveTo(poly[0])
                    for p in poly[1:]: path.lineTo(p) # Create open path
                    self.temp_item = self.scene().addPath(path, QPen(Qt.gray, 0, Qt.DashLine))
                elif self.temp_path_points: # Show first point marker
                    marker_size = max(2, self.parent_win.project.grid_pitch / 10 * self.transform().m11())
                    marker_rect = QRectF(scene_pos - QPointF(marker_size/2, marker_size/2), QSizeF(marker_size, marker_size))
                    self.temp_item = self.scene().addRect(marker_rect, QPen(Qt.gray, 0), QBrush(Qt.gray))
            else:
                super().mousePressEvent(event) # Pass to base class for selection/move
                return
            event.accept()
        else:
            super().mousePressEvent(event) # Handle other buttons (e.g., right-click context menu)


    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        menu = QMenu(self)

        paste_action = menu.addAction("Paste")
        paste_action.triggered.connect(lambda: self.parent_win.paste_shapes(self.mapToScene(event.pos())))
        if not self.parent_win.can_paste(): paste_action.setEnabled(False)

        if isinstance(item, (RectItem, PolyItem, CircleItem, RefItem)):
            if not item.isSelected():
                # Allow selecting multiple items before right-clicking
                # Only clear if the clicked item wasn't already part of the selection
                if not item in self.scene().selectedItems():
                    self.scene().clearSelection()
                    item.setSelected(True)


            menu.addSeparator()
            prop = menu.addAction("Properties..."); prop.triggered.connect(lambda: self.parent_win.show_properties_dialog_for_item(item))
            rename = menu.addAction("Rename..."); rename.triggered.connect(self.parent_win.rename_selected_shape)
            measure = menu.addAction("Measure..."); measure.triggered.connect(self.parent_win.measure_selection)

            menu.addSeparator()
            copy = menu.addAction("Copy"); copy.triggered.connect(self.parent_win.copy_selected_shapes)

            arrange_menu = menu.addMenu("Arrange")
            front = arrange_menu.addAction("Bring to Front"); front.triggered.connect(self.parent_win.bring_selection_to_front)
            forward = arrange_menu.addAction("Move Forward"); forward.triggered.connect(self.parent_win.move_selection_forward)
            backward = arrange_menu.addAction("Move Backward"); backward.triggered.connect(self.parent_win.move_selection_backward)
            back = arrange_menu.addAction("Send to Back"); back.triggered.connect(self.parent_win.send_selection_to_back)

            transform_menu = menu.addMenu("Transform")
            rot = transform_menu.addAction("Rotate..."); rot.triggered.connect(self.parent_win.rotate_selection)
            sca = transform_menu.addAction("Scale..."); sca.triggered.connect(self.parent_win.scale_selection)
            flip_menu = transform_menu.addMenu("Flip")
            fh = flip_menu.addAction("Horizontal"); fh.triggered.connect(self.parent_win.flip_selection_horizontal)
            fv = flip_menu.addAction("Vertical"); fv.triggered.connect(self.parent_win.flip_selection_vertical)

            move_menu = menu.addMenu("Move to Layer")
            for layer in self.parent_win.project.layers:
                layer_action = move_menu.addAction(layer.name)
                # Use lambda capture to get the correct layer name
                layer_action.triggered.connect(lambda checked=False, l_name=layer.name: self.parent_win.move_selection_to_layer(l_name))


            menu.addSeparator()
            delete = menu.addAction("Delete"); delete.triggered.connect(self.parent_win.delete_selected_items)

        menu.exec_(event.globalPos())

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        # Always update status bar, even if panning
        self.parent_win.statusBar().showMessage(f"X: {scene_pos.x():.2f}, Y: {-scene_pos.y():.2f}") # Y is inverted

        if self._panning:
            delta = event.pos() - self._pan_start ; self._pan_start = event.pos()
            # Use scrollbars for panning
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept(); return

        # Drawing Temp Items
        if self.temp_item and self.start_pos:
            snapped_pos = self.snap_point(scene_pos)
            if self.mode in [self.MODES["rect"], self.MODES["circle"]]:
                self.temp_item.setRect(QRectF(self.start_pos, snapped_pos).normalized())
            elif self.mode == self.MODES["measure"]:
                line = QLineF(self.start_pos, snapped_pos)
                self.temp_item.setLine(line)
                # Update status bar with measure info
                self.parent_win.statusBar().showMessage(f"dX: {line.dx():.2f}, dY: {-line.dy():.2f}, Dist: {line.length():.2f}")
            event.accept(); return
        elif self.mode == self.MODES["poly"] and len(self.temp_poly_points) > 0:
            if self.temp_item: self.scene().removeItem(self.temp_item) # Remove previous temp line/poly
            poly_pts = self.temp_poly_points + [self.snap_point(scene_pos)] # Add current mouse pos
            poly = QPolygonF(poly_pts)
            self.temp_item = self.scene().addPolygon(poly, QPen(Qt.gray, 0, Qt.DashLine))
            event.accept(); return
        elif self.mode == self.MODES["path"] and len(self.temp_path_points) > 0:
            if self.temp_item: self.scene().removeItem(self.temp_item)
            path_pts = self.temp_path_points + [self.snap_point(scene_pos)]
            poly = QPolygonF(path_pts)
            path = QPainterPath(); path.moveTo(poly[0])
            for p in poly[1:]: path.lineTo(p) # Create open path
            self.temp_item = self.scene().addPath(path, QPen(Qt.gray, 0, Qt.DashLine))
            event.accept(); return

        super().mouseMoveEvent(event) # Pass to base class for selection rubber band / item moving


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False; self.setCursor(Qt.ArrowCursor); event.accept(); return
        if self.start_pos and self.mode in [self.MODES["rect"], self.MODES["circle"]]:
            rect = self.temp_item.rect()
            self.temp_cancel() # Remove temp item BEFORE adding permanent one
            if rect.width() > 0 and rect.height() > 0 and (layer := self.get_active_layer()):
                if self.mode == self.MODES["rect"]: self.parent_win.add_rect_to_active_cell(rect, layer)
                else: self.parent_win.add_circle_to_active_cell(rect, layer)
            event.accept(); return
        if self.start_pos and self.mode == self.MODES["measure"]:
            self.temp_cancel(); event.accept(); return # Just remove the measurement line
        # No special release action for poly/path needed here, happens on double-click or escape
        super().mouseReleaseEvent(event) # Pass to base class


    def mouseDoubleClickEvent(self, event):
        if self.mode == self.MODES["path"] and event.button() == Qt.LeftButton:
            self.finish_path(); event.accept(); return
        if self.mode == self.MODES["poly"] and event.button() == Qt.LeftButton:
            self.finish_polygon(); event.accept(); return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.parent_win.set_active_tool("select") # Also cancels temp items via set_mode
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
             # Finish poly/path on Enter key
             if self.mode == self.MODES["poly"]:
                 self.finish_polygon()
             elif self.mode == self.MODES["path"]:
                 self.finish_path()
        else:
            super().keyPressEvent(event)


    def finish_polygon(self):
        can_create = len(self.temp_poly_points) >= 3 and (layer := self.get_active_layer())
        if can_create: poly = QPolygonF(self.temp_poly_points)
        self.temp_cancel() # Clear temp items
        if can_create: self.parent_win.add_poly_to_active_cell(poly, layer)

    def finish_path(self):
        can_create = len(self.temp_path_points) >= 2 and (layer := self.get_active_layer())
        if can_create:
            points = self.temp_path_points[:] # Make a copy before clearing
        else:
            points = []
        self.temp_cancel()
        if can_create:
            self.parent_win.add_path_to_active_cell(points, layer)

    def dragEnterEvent(self, event): event.acceptProposedAction() if event.mimeData().hasText() else event.ignore()
    def dragMoveEvent(self, event): event.acceptProposedAction() if event.mimeData().hasText() else event.ignore()

    def dropEvent(self, event):
        cell_name = event.mimeData().text()
        if not (cell_name and cell_name in self.parent_win.project.cells):
            event.ignore(); return
        if cell_name == self.parent_win.active_cell_name:
            QToolTip.showText(event.globalPos(), "Cannot drop a cell onto itself."); event.ignore(); return

        item_at_pos = self.itemAt(event.pos())
        target_data_obj = None

        # Check if dropped onto a selectable shape item
        if isinstance(item_at_pos, (RectItem, PolyItem, CircleItem)) and \
           (item_at_pos.flags() & QGraphicsItem.ItemIsSelectable):
            target_data_obj = item_at_pos.data_obj

        if target_data_obj:
            self.parent_win.instantiate_cell_on_shape(cell_name, target_data_obj)
        else:
            scene_pos = self.mapToScene(event.pos())
            self.parent_win.add_reference_to_active_cell(cell_name, self.snap_point(scene_pos))

        event.acceptProposedAction()

from PyQt5.QtCore import QThread, pyqtSignal

# --- 1. The Worker Thread (Does the math in background) ---
class SimulationWorker(QThread):
    progress_update = pyqtSignal(int, str)
    finished_data = pyqtSignal(object, float, tuple) # Added tuple for (min_x, min_y) offset
    error_occurred = pyqtSignal(str)

    def __init__(self, project, width, height):
        super().__init__()
        self.project = project
        # We ignore width/height inputs now and calculate them dynamically

    def run(self):
        try:
            import numpy as np
            from scipy.ndimage import distance_transform_edt, gaussian_filter
            import matplotlib.path as mpath

            self.progress_update.emit(5, "Analyzing Layout...")

            # --- 1. Dynamic Bounding Box Calculation ---
            # Collect all points from all visible polygons to find the active area
            all_xs = []
            all_ys = []
            cell = self.project.cells[self.project.top]
            
            visible_layers = [l for l in self.project.layers if l.visible]
            visible_names = {l.name for l in visible_layers}

            has_content = False
            for p in cell.polygons:
                if p.layer in visible_names and p.points:
                    xs = [pt[0] for pt in p.points]
                    ys = [pt[1] for pt in p.points]
                    all_xs.extend(xs)
                    all_ys.extend(ys)
                    has_content = True
            
            # Also check ellipses (approximate bounding box)
            for e in cell.ellipses:
                if e.layer in visible_names:
                    all_xs.extend([e.rect[0], e.rect[0] + e.rect[2]])
                    all_ys.extend([e.rect[1], e.rect[1] + e.rect[3]])
                    has_content = True

            if not has_content:
                self.error_occurred.emit("No visible shapes found to simulate.")
                return

            # Define bounds with a margin
            margin = 20.0
            min_x, max_x = min(all_xs) - margin, max(all_xs) + margin
            min_y, max_y = min(all_ys) - margin, max(all_ys) + margin
            
            real_w = max_x - min_x
            real_h = max_y - min_y

            # --- 2. Smart Resolution ---
            # Fixed grid size of 256 on the longest side for performance
            # This makes small chips look detailed and huge chips process fast.
            target_dim = 256
            res = max(real_w, real_h) / target_dim
            
            w = int(real_w / res)
            h = int(real_h / res)
            d = 60 # Z-depth resolution (layers)

            self.progress_update.emit(10, f"Grid Size: {w}x{h}x{d}")

            volume = np.zeros((w, h, d), dtype=np.uint8)

            # --- 3. Generate Coordinate Grid ---
            # Create a grid that exactly matches the bounded area
            x_space = np.linspace(min_x, max_x, w)
            y_space = np.linspace(min_y, max_y, h)
            xv, yv = np.meshgrid(x_space, y_space)
            # Flatten for vectorized containment check
            grid_points = np.vstack((xv.flatten(), yv.flatten())).T

            # --- 4. Substrate Initialization ---
            sub_h = int(d * 0.15) # Bottom 15% is substrate
            volume[:, :, 0:sub_h] = 1 
            current_z = sub_h

            # --- 5. Process Layers ---
            mat_id = 2
            total_layers = len(visible_layers)

            for i, layer in enumerate(visible_layers):
                msg = f"Processing: {layer.name}"
                self.progress_update.emit(15 + int((i/total_layers)*80), msg)

                # Collect polygons for THIS layer
                polys = [p.points for p in cell.polygons if p.layer == layer.name]
                
                # Handle Ellipses for this layer (Convert to Path)
                
                ellipses = [e for e in cell.ellipses if e.layer == layer.name]
                for e in ellipses:
                    # Convert ellipse to polygon points (32 segments)
                    cx, cy = e.rect[0] + e.rect[2]/2, e.rect[1] + e.rect[3]/2
                    rx, ry = e.rect[2]/2, e.rect[3]/2
                    theta = np.linspace(0, 2*np.pi, 32)
                    ex = cx + rx * np.cos(theta)
                    ey = cy + ry * np.sin(theta)
                    pts = np.column_stack([ex, ey])
                    polys.append(pts) # Add to the list to be processed
                # --------------------------------------------------

                if not polys: continue
                # ... (Simplified: Ellipses handled via bounding box above, 
                # strict rasterization requires converting ellipse to polygon points here.
                # For now, we rely on the user converting circles to polys or standard polys)

                # Rasterize
                layer_mask_flat = np.zeros(w*h, dtype=bool)
                
                for p_points in polys:
                    path = mpath.Path(p_points)
                    # Determine which grid points are inside this polygon
                    mask = path.contains_points(grid_points)
                    layer_mask_flat |= mask
                
                layer_mask = layer_mask_flat.reshape(h, w).T # Reshape back to grid

                if not np.any(layer_mask): continue

                # Realism: Blur/Round
                mask_float = layer_mask.astype(float)
                mask_blurred = gaussian_filter(mask_float, sigma=1.0)
                mask_processed = mask_blurred > 0.5

                # Realism: Taper
                dt = distance_transform_edt(mask_processed)
                
                # Thickness in voxels
                thick_vox = max(1, int(layer.thickness_2_5d / res))
                taper_angle = 10.0

                for z in range(thick_vox):
                    z_idx = current_z + z
                    if z_idx >= d: break
                    
                    erosion = z * np.tan(np.radians(taper_angle))
                    # Solid if distance from edge > erosion amount
                    z_mask = dt > erosion
                    
                    # Overwrite existing voxels
                    volume[z_mask, z_idx] = mat_id

                current_z += thick_vox
                mat_id += 1

            self.progress_update.emit(100, "Meshing...")
            # Pass back volume, resolution, and the top-left corner offset
            self.finished_data.emit(volume, res, (min_x, min_y))

        except Exception as e:
            self.error_occurred.emit(str(e))


class ThreeDViewDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive 3D Process Viewer")
        self.resize(1400, 900)
        self.project = project
        self.worker = None
        self.generated_meshes = []  # Store references to meshes for live updates
        self.current_z_scale = 1.0

        # --- Layouts ---
        main_layout = QHBoxLayout(self)  # Side-by-side layout

        # 1. Left Side: The 3D Viewport
        if _has_3d_deps:
            self.gl_view = gl.GLViewWidget()
            self.gl_view.opts['distance'] = 200  # Initial camera distance
            self.gl_view.opts['azimuth'] = -45
            self.gl_view.opts['elevation'] = 30
            self.gl_view.setBackgroundColor('#1e1e1e')  # Dark Gray vs Pitch Black

            # Grid and Axis
            self.grid_item = gl.GLGridItem()
            self.grid_item.scale(20, 20, 1)
            self.gl_view.addItem(self.grid_item)

            self.axis_item = gl.GLAxisItem()
            self.axis_item.setSize(50, 50, 50)
            self.gl_view.addItem(self.axis_item)

            main_layout.addWidget(self.gl_view, stretch=4)
        else:
            main_layout.addWidget(QLabel("Error: Missing pyqtgraph/skimage"))

        # 2. Right Side: Control Panel
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setMaximumWidth(300)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(15)

        # -- Section: Simulation --
        panel_layout.addWidget(QLabel("<b>1. Simulation</b>"))
        self.lbl_status = QLabel("Ready to build.")
        self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
        panel_layout.addWidget(self.lbl_status)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        panel_layout.addWidget(self.progress)

        # Store as attribute so we can enable/disable it
        self.btn_run = QPushButton("Generate 3D Model")
        self.btn_run.setStyleSheet(
            "background-color: #007ACC; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_run.clicked.connect(self.start_simulation)
        panel_layout.addWidget(self.btn_run)

        # NEW: Reset button
        self.btn_reset = QPushButton("Reset 3D View")
        self.btn_reset.clicked.connect(self.reset_view)
        panel_layout.addWidget(self.btn_reset)

        panel_layout.addWidget(self._create_separator())

        # -- Section: Camera & View --
        panel_layout.addWidget(QLabel("<b>2. Camera Presets</b>"))
        cam_grid = QGridLayout()
        btn_iso = QPushButton("ISO")
        btn_iso.clicked.connect(lambda: self.set_cam(-45, 30))
        btn_top = QPushButton("TOP")
        btn_top.clicked.connect(lambda: self.set_cam(0, 90))
        btn_front = QPushButton("FRONT")
        btn_front.clicked.connect(lambda: self.set_cam(-90, 0))
        btn_side = QPushButton("SIDE")
        btn_side.clicked.connect(lambda: self.set_cam(0, 0))
        cam_grid.addWidget(btn_iso, 0, 0)
        cam_grid.addWidget(btn_top, 0, 1)
        cam_grid.addWidget(btn_front, 1, 0)
        cam_grid.addWidget(btn_side, 1, 1)
        panel_layout.addLayout(cam_grid)

        # -- Section: Appearance --
        panel_layout.addWidget(QLabel("<b>3. Appearance</b>"))

        # Z-Scale Slider
        panel_layout.addWidget(QLabel("Z-Axis Exaggeration:"))
        self.slider_z = QSlider(Qt.Horizontal)
        self.slider_z.setRange(1, 200)  # 0.1x to 20.0x
        self.slider_z.setValue(20)      # Default 2.0x
        self.slider_z.valueChanged.connect(self.update_z_scale)
        panel_layout.addWidget(self.slider_z)

        # Transparency
        self.chk_translucent = QCheckBox("Translucent Materials")
        self.chk_translucent.setChecked(True)
        self.chk_translucent.toggled.connect(self.toggle_transparency)
        panel_layout.addWidget(self.chk_translucent)

        # Grid Toggle
        chk_grid = QCheckBox("Show Grid/Axis")
        chk_grid.setChecked(True)
        chk_grid.toggled.connect(self.toggle_grid)
        panel_layout.addWidget(chk_grid)

        panel_layout.addWidget(self._create_separator())

        # -- Section: Export --
        btn_snap = QPushButton("Save Screenshot")
        btn_snap.clicked.connect(self.save_snapshot)
        panel_layout.addWidget(btn_snap)

        panel_layout.addStretch()
        main_layout.addWidget(panel, stretch=1)

        # Materials colors
        self.colors = [
            (0, 0, 0, 0),
            (0.6, 0.6, 0.6, 1),
            (0.2, 0.4, 0.8, 0.6),
            (0.8, 0.2, 0.2, 0.6),
            (0.2, 0.8, 0.2, 0.6),
            (0.9, 0.9, 0.2, 0.7),
        ]


    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def set_cam(self, az, el):
        self.gl_view.setCameraPosition(elevation=el, azimuth=az)

    def toggle_grid(self, visible):
        self.grid_item.setVisible(visible)
        self.axis_item.setVisible(visible)

    def toggle_transparency(self, checked):
        """
        Toggle between translucent and solid (opaque) materials.

        When checked  -> use original alpha and 'translucent' blending.
        When unchecked -> force alpha = 1.0 and use 'opaque' blending.
        """
        if not hasattr(self, "generated_meshes") or not self.generated_meshes:
            return

        for mesh_item in self.generated_meshes:
            # Base color was stored when meshes were created
            base = getattr(mesh_item, "_base_color", (1.0, 1.0, 1.0, 1.0))
            r, g, b, a = base

            if checked:
                # Translucent view: use original alpha and translucent blending
                mesh_item.setGLOptions("translucent")
                mesh_item.setColor((r, g, b, a))
            else:
                # Solid view: alpha = 1 and opaque blending
                mesh_item.setGLOptions("opaque")
                mesh_item.setColor((r, g, b, 1.0))

        if hasattr(self, "gl_view"):
            self.gl_view.update()


    def update_z_scale(self):
        # Convert slider 1-200 to float 0.1-20.0
        new_scale = self.slider_z.value() / 10.0
        
        if not self.generated_meshes: return
        
        # Apply transform relative to previous scale
        for item in self.generated_meshes:
            item.resetTransform() # Clear old scaling
            # Scale X,Y=1 (normal), Z=new_scale
            item.scale(1, 1, new_scale) 
            
        self.current_z_scale = new_scale

    def save_snapshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "3d_view.png", "PNG (*.png)")
        if path:
            self.gl_view.grabFrameBuffer().save(path)

    def start_simulation(self):
        # Show progress bar + reset it
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.lbl_status.setText("Running 3D process simulation...")

        # Disable the run button while simulation is in progress
        if hasattr(self, "btn_run"):
            self.btn_run.setEnabled(False)

        # Stop any previous worker if it is still running
        if self.worker is not None and self.worker.isRunning():
            try:
                self.worker.terminate()
                self.worker.wait()
            except Exception:
                pass
        self.worker = None

        # Clear old meshes
        for m in self.generated_meshes:
            try:
                self.gl_view.removeItem(m)
            except Exception:
                pass
        self.generated_meshes.clear()

        # Start Worker
        self.worker = SimulationWorker(
            self.project, self.project.canvas_width, self.project.canvas_height
        )
        self.worker.progress_update.connect(self.on_progress)
        self.worker.finished_data.connect(self.on_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()


    def on_progress(self, percent, message):
        self.progress.setValue(percent)
        self.lbl_status.setText(message)

    def on_error(self, msg):
        # Called from the worker thread when something goes wrong
        self.worker = None
        self.lbl_status.setText("Error!")
        QMessageBox.critical(self, "Simulation Error", msg)
        self.progress.setVisible(False)

        # Re-enable the run button so user can try again
        if hasattr(self, "btn_run"):
            self.btn_run.setEnabled(True)

    def reset_view(self):
        """
        Completely reset 3D view, state, and UI so previous runs/errors
        don't carry over.
        """

        # Stop any running worker safely
        if self.worker is not None and self.worker.isRunning():
            try:
                self.worker.terminate()
                self.worker.wait()
            except Exception:
                pass
        self.worker = None

        # Remove all generated meshes from the GL view
        if hasattr(self, "gl_view"):
            for m in self.generated_meshes:
                try:
                    self.gl_view.removeItem(m)
                except Exception:
                    pass
        self.generated_meshes.clear()

        # Reset camera to default
        if hasattr(self, "gl_view"):
            self.gl_view.opts["distance"] = 200
            self.gl_view.opts["azimuth"] = -45
            self.gl_view.opts["elevation"] = 30
            self.gl_view.update()

        # Reset Z-scale (back to default slider value)
        self.current_z_scale = 1.0
        if hasattr(self, "slider_z"):
            self.slider_z.blockSignals(True)
            self.slider_z.setValue(20)  # same as initial value in __init__
            self.slider_z.blockSignals(False)
            # No need to call update_z_scale() because there are no meshes yet

        # Make sure grid / axis and translucency are back on
        if hasattr(self, "grid_item"):
            self.grid_item.setVisible(True)
        if hasattr(self, "axis_item"):
            self.axis_item.setVisible(True)
        if hasattr(self, "chk_translucent"):
            self.chk_translucent.setChecked(True)

        # Reset status/progress text
        self.progress.setVisible(False)
        self.progress.setValue(0)
        self.lbl_status.setText("Ready to build.")

        # Ensure Run button is usable again
        if hasattr(self, "btn_run"):
            self.btn_run.setEnabled(True)
    
    def on_finished(self, volume, res, offset):
        """
        Called when the SimulationWorker finishes.

        volume : 3D numpy array of material IDs
        res    : in-plane resolution (x/y step in world units)
        offset : (min_x, min_y) real-world origin for this cropped volume
        """
        self.lbl_status.setText("Rendering 3D model...")
        QApplication.processEvents()

        min_x, min_y = offset

        try:
            unique_mats = np.unique(volume)

            # Compute center of model in world coordinates for nicer centering
            size_x = volume.shape[0] * res
            size_y = volume.shape[1] * res
            center_x = min_x + size_x / 2.0
            center_y = min_y + size_y / 2.0

            # Remove any old meshes just in case
            if hasattr(self, "generated_meshes"):
                for m in self.generated_meshes:
                    try:
                        self.gl_view.removeItem(m)
                    except Exception:
                        pass
                self.generated_meshes.clear()
            else:
                self.generated_meshes = []

            for mid in unique_mats:
                if mid == 0:
                    continue  # 0 = empty

                # 1) Isolate this material as a float volume
                mat_vol = (volume == mid).astype(float)

                # 2) Slight smoothing before marching cubes
                mat_vol = gaussian_filter(mat_vol, sigma=0.5)

                # 3) Marching Cubes
                verts, faces, normals, _ = measure.marching_cubes(mat_vol, level=0.5)

                # 4) Convert from voxel coords to world coords
                #    x -> i * res + min_x, y -> j * res + min_y, z -> k * (res or 1)
                verts[:, 0] = verts[:, 0] * res + min_x
                verts[:, 1] = verts[:, 1] * res + min_y
                verts[:, 2] = verts[:, 2] * res  # or 1.0 if your z spacing differs

                # 5) Center model around origin for nicer camera handling
                verts[:, 0] -= center_x
                verts[:, 1] -= center_y

                # Pick a color for this material
                base_color = self.colors[mid % len(self.colors)]

                # Decide initial glOptions based on checkbox state
                if self.chk_translucent.isChecked():
                    gl_opts = "translucent"
                else:
                    gl_opts = "opaque"
                    # If solid mode, force alpha to 1
                    if len(base_color) == 4:
                        base_color = (base_color[0], base_color[1],
                                      base_color[2], 1.0)

                # 6) Create GLMeshItem
                mesh = gl.GLMeshItem(
                    vertexes=verts,
                    faces=faces,
                    normals=normals,
                    color=base_color,
                    smooth=True,
                    glOptions=gl_opts,
                )

                # Store base color so toggle_transparency can restore it
                mesh._base_color = base_color

                # Add to scene and track it
                self.gl_view.addItem(mesh)
                self.generated_meshes.append(mesh)

            # Apply current Z scale to the new meshes (if you have that logic)
            if hasattr(self, "update_z_scale"):
                self.update_z_scale()

            self.lbl_status.setText("Done.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.lbl_status.setText("Render Failed.")
            QMessageBox.critical(self, "Render Error", str(e))

        # Worker finished; clean up
        self.worker = None
        self.progress.setVisible(False)

        # Re-enable Run button
        if hasattr(self, "btn_run"):
            self.btn_run.setEnabled(True)


    

# -------------------------- Main window --------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = None
        self.current_file_path = None
        self._is_dirty = False
        self._undo_stack, self._redo_stack = [], []
        self._build_ui()
        self._apply_stylesheet()
        self.setCentralWidget(QLabel("Create a new project or open a file to begin.", alignment=Qt.AlignCenter))
        self._update_window_title()
        self.statusBar().showMessage("Welcome!")
        self.resize(1600, 1000)

    def show_help_pdf(self):
        """Finds and opens the user guide PDF."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        pdf_path = os.path.join(script_dir, 'docs', 'user_guide.pdf')

        if os.path.exists(pdf_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path))
        else:
            QMessageBox.warning(self, "Help Not Found",
                                f"Could not find the user guide.\n"
                                f"Expected location: {pdf_path}")

    def initialize_project(self, project):
        self.project = project
        
        # Ensure top cell exists, fallback if not
        if not self.project.top or self.project.top not in self.project.cells:
             if self.project.cells:
                 self.project.top = list(self.project.cells.keys())[0]
             else: # No cells exist at all, create a default TOP
                 self.project.top = "TOP"
                 self.project.cells["TOP"] = Cell()
                 
        self.active_cell_name = self.project.top
        self.active_layer_name = self.project.layers[0].name if self.project.layers else None
        
        self.scene = QGraphicsScene() # Use a boundless scene
        self.view = Canvas(self.scene, self.active_layer, self)
        self.view.scale(1, -1) # Invert Y axis for typical layout coordinates
        self.canvas_container = CanvasContainer(self.view, self)
        self.setCentralWidget(self.canvas_container)
        
        self.update_ui_from_project() # This calls _redraw_scene() which populates the scene
        
        self._undo_stack, self._redo_stack = [], []
        self._save_state() # Initial state
        self._is_dirty = False
        self._update_window_title()
        self.fill_view()

    def _create_default_project(self, pitch, width, height):
        layers = [Layer("N-doping", (0, 0, 255)), Layer("P-doping", (255, 0, 0)), Layer("Oxide", (0, 200, 0)), Layer("Metal", (200, 200, 200)), Layer("Contact", (255, 200, 0))]
        return Project(layers=layers, cells={"TOP": Cell()}, top="TOP", grid_pitch=pitch, canvas_width=width, canvas_height=height)

    def _mark_dirty(self):
        if not self._is_dirty: self._is_dirty = True; self._update_window_title()

    def _update_window_title(self):
        filename = os.path.basename(self.current_file_path) if self.current_file_path else "Untitled"
        self.setWindowTitle(f"{filename}[*] - 2D Mask Layout Editor")
        self.setWindowModified(self._is_dirty)

    def closeEvent(self, event):
        if self._is_dirty:
            reply = QMessageBox.question(self, 'Unsaved Changes', "You have unsaved changes. Do you want to save them before closing?",
                                         QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if reply == QMessageBox.Save:
                if not self.save_json(): event.ignore()
            elif reply == QMessageBox.Cancel:
                event.ignore()
            # If Discard, proceed to close (accept event)

    def _prompt_save_if_dirty(self):
        if not self._is_dirty: return True
        reply = QMessageBox.question(self, 'Unsaved Changes', "You have unsaved changes. Do you want to save them first?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if reply == QMessageBox.Save: return self.save_json()
        return reply != QMessageBox.Cancel

    def new_doc(self):
        if not self._prompt_save_if_dirty(): return
        grid_dialog = GridDialog(self)
        if grid_dialog.exec_() == QDialog.Accepted:
            pitch, w, h = grid_dialog.get_values()
            self.current_file_path = None
            self.initialize_project(self._create_default_project(pitch, w, h))

    def open_file(self, path=None):
        if not self._prompt_save_if_dirty(): return
        if not path:
            filters = "Layout Files (*.json *.gds *.oas);;JSON Project (*.json);;GDSII Files (*.gds);;OASIS Files (*.oas);;All Files (*)"
            path, _ = QFileDialog.getOpenFileName(self, "Open File", "", filters)
        if not path: return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.json': self._load_json(path)
            elif ext in ['.gds', '.oas']: self._load_gds_oas(path)
            else: raise ValueError(f"Unsupported file type: {ext}")
            self.statusBar().showMessage(f"Successfully opened {path}")
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to open file:\n{e}")
            self.statusBar().showMessage(f"Failed to open {path}")
        finally:
            QApplication.restoreOverrideCursor()

    def _load_json(self, path):
        with open(path, "r") as f: data = json.load(f)
        
        # --- Make sure layer has new attributes ---
        loaded_layers = []
        for l_data in data.get('layers', []):
            if 'fill_pattern' not in l_data: l_data['fill_pattern'] = 'solid'
            if 'thickness_2_5d' not in l_data: l_data['thickness_2_5d'] = 10.0
            loaded_layers.append(Layer(**l_data))

        project = Project(
            units=data.get('units', 'um'), top=data.get('top', 'TOP'),
            layers=loaded_layers, # Use the potentially updated list
            grid_pitch=data.get('grid_pitch', 50),
            canvas_width=data.get('canvas_width', 2000), canvas_height=data.get('canvas_height', 1500)
        )
        for name, c_data in data.get('cells', {}).items():
            # Handle potential missing UUIDs on load more gracefully
            polys = []
            for p_dict in c_data.get('polygons', []):
                p_dict.pop('uuid', None) # Always remove old uuid if present
                polys.append(Poly(**p_dict))

            ellipses = []
            for e_dict in c_data.get('ellipses', []):
                e_dict.pop('uuid', None)
                ellipses.append(Ellipse(**e_dict))

            refs = [Ref(**r) for r in c_data.get('references', [])]
            # Load etch pairs if they exist
            etch_pairs = [tuple(ep) for ep in c_data.get('etch_pairs', [])]
            project.cells[name] = Cell(polygons=polys, ellipses=ellipses, references=refs, etch_pairs=etch_pairs)

        project.refresh_layer_map()
        self.current_file_path = path
        self.initialize_project(project)

    def _load_gds_oas(self, path):
        if not gdstk: raise ImportError("GDSTK library is required for this feature.")
        unit_microns = 1e-6
        lib = gdstk.read_gds(path, unit=unit_microns) if path.lower().endswith('.gds') else gdstk.read_oas(path, unit=unit_microns)

        sidecar_path = path + '.json'
        layer_meta = {}
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, 'r') as f:
                    layer_meta = {int(k): v for k, v in json.load(f).get('layer_metadata', {}).items()}
            except Exception as e: print(f"Warning: Could not read sidecar metadata file. {e}")
        
        project = self._create_default_project(50, 2000, 1500)
        project.cells.clear() # Start with fresh cells/layers from default
        project.layers.clear() # Clear default layers

        gds_layers = sorted(list(set(ld[0] for ld in lib.layers_and_datatypes()))) # Unique layer numbers
        layer_num_to_name = {}
        existing_layer_names = set()

        for l_num in gds_layers:
            layer_name = None
            if l_num in layer_meta:
                meta = layer_meta[l_num]
                layer_name = meta.get('name')
                if layer_name and layer_name not in existing_layer_names:
                    project.layers.append(Layer(
                        name=layer_name, 
                        color=tuple(meta.get('color', (l_num*20%255, l_num*50%255, l_num*80%255))), 
                        visible=meta.get('visible', True),
                        fill_pattern=meta.get('fill_pattern', 'solid'),
                        thickness_2_5d=meta.get('thickness_2_5d', 10.0)
                        ))
                    existing_layer_names.add(layer_name)
            
            if not layer_name or layer_name in existing_layer_names and l_num not in layer_num_to_name:
                # If meta name was missing, or was a duplicate, create a default name
                layer_name = f"Layer_{l_num}"
                if layer_name not in existing_layer_names:
                    project.layers.append(Layer(
                        layer_name, 
                        (l_num*20%255, l_num*50%255, l_num*80%255),
                        fill_pattern='solid',
                        thickness_2_5d=10.0
                        ))
                    existing_layer_names.add(layer_name)
            
            layer_num_to_name[l_num] = layer_name # Map GDS layer number to our name


        project.refresh_layer_map()
        
        for cell in lib.cells:
            new_cell = Cell()
            for poly in cell.polygons:
                # Find the layer name matching the GDS layer number
                layer_name = layer_num_to_name.get(poly.layer)
                # If layer was defined, add the polygon
                if layer_name and layer_name in project.layer_by_name:
                    new_cell.polygons.append(Poly(layer_name, poly.points.tolist()))
                # Optional: Handle datatypes if needed, e.g., f"Layer_{poly.layer}_{poly.datatype}"
                # else:
                #    print(f"Warning: Skipping polygon on unmapped GDS layer {poly.layer}")

            for ref in cell.references:
                ref_cell_name = ref.cell.name if isinstance(ref.cell, gdstk.Cell) else ref.cell
                new_cell.references.append(Ref(ref_cell_name, ref.origin, ref.rotation, ref.magnification))
            project.cells[cell.name] = new_cell
            
        # Robustly find and set the top cell
        if lib.top_level(): 
            project.top = lib.top_level()[0].name
        elif project.cells: 
            project.top = list(project.cells.keys())[0]
        else:
            project.top = "TOP"
            project.cells["TOP"] = Cell()

        self.current_file_path = None # Don't overwrite GDS
        self.initialize_project(project)

    def save_json(self):
        return self.save_json_as() if not self.current_file_path else self._write_json(self.current_file_path)

    def save_json_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON As", self.current_file_path or "", "JSON Files (*.json)")
        return self._write_json(path) if path else False

    def _write_json(self, path):
        if not self.project: return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with open(path, "w") as f:
                json.dump(self.project, f, indent=2, cls=ProjectEncoder)
            self.current_file_path, self._is_dirty = path, False
            self._update_window_title()
            self.statusBar().showMessage(f"Saved to {path}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{e}")
            self.statusBar().showMessage(f"Failed to save to {path}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _write_gdstk(self, is_oas):
        if not gdstk: QMessageBox.warning(self, "Feature Disabled", "Install 'gdstk'."); return
        ext = "*.oas" if is_oas else "*.gds"
        path, _ = QFileDialog.getSaveFileName(self, f"Save {ext.upper()}", "", f"Files ({ext})")
        if not path: return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            units_in_meters = {'um': 1e-6, 'nm': 1e-9, 'mm': 1e-3}
            unit_value = units_in_meters.get(self.project.units.lower(), 1e-6) # Defaults to um
            lib = gdstk.Library(unit=unit_value) 
            layer_map = {layer.name: i for i, layer in enumerate(self.project.layers, 1)}
            gds_cells = {name: lib.new_cell(name) for name in self.project.cells}
            for name, cell in self.project.cells.items():
                gds_cell = gds_cells[name]
                for p in cell.polygons: gds_cell.add(gdstk.Polygon(p.points, layer=layer_map.get(p.layer, 0)))
                for e in cell.ellipses:
                    r = e.rect; center = (r[0]+r[2]/2, r[1]+r[3]/2); 
                    radius = (r[2] + r[3]) / 4 # Average radius for ellipse approx
                    gds_cell.add(gdstk.ellipse(center, radius, layer=layer_map.get(e.layer, 0)))
                for r in cell.references:
                    if r.cell in gds_cells:
                        gds_cell.add(gdstk.Reference(gds_cells[r.cell], r.origin, r.rotation, r.magnification))
            
            if is_oas: lib.write_oas(path)
            else: lib.write_gds(path)
            
            # Save metadata including fill_pattern and thickness
            meta = {layer_map[l.name]: {
                        'name': l.name, 
                        'color': l.color, 
                        'visible': l.visible,
                        'fill_pattern': l.fill_pattern,
                        'thickness_2_5d': getattr(l, 'thickness_2_5d', 10.0) # Save thickness
                    } for l in self.project.layers if l.name in layer_map}
            
            with open(path + '.json', 'w') as f: json.dump({'layer_metadata': meta}, f, indent=2)
            self.statusBar().showMessage(f"Exported to {path} and created metadata file.")
        except Exception as e: QMessageBox.critical(self, "Export Error", f"Failed to write file: {e}")
        finally: QApplication.restoreOverrideCursor()

    # UPDATED to handle new ProcessDialog
    def run_process_step(self):
        """Opens the process simulation dialog and applies the selected fabrication step."""
        if not self.project:
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        dlg = ProcessDialog(self.project, self)
        if dlg.exec_() != QDialog.Accepted:
            return

        values = dlg.get_values()
        step = values["step"]
        ptype = values["type"]
        layer1 = values["layer1"]
        layer2 = values["layer2"]
        param1 = values["param1"]
        param2 = values["param2"]

        # --- Dispatching ---
        if step == "Deposition":
            # Option 1: Basic thickness update for 2.5D view
            self._apply_deposition(layer1, param1, True)
            # Option 2: Trigger Level Set (adds to new layer)
            # Note: speed needs interpretation from thickness/time
            # self.run_level_set_simulation(
            #     target_layer_name=layer1, process_type="deposit",
            #     duration=1.0, speed=param1 if param1 else 10.0, # Example mapping
            #     grid_resolution=self.project.grid_pitch / 10.0
            # )

        elif step == "Etch":
            if layer1 and layer2:
                 # 1. Update etch pairs for 2.5D view and set mask style
                self._apply_etching(layer1, layer2, param2 if param2 is not None else 100) # Use depth (param2)
                # 2. Trigger Level Set (adds to new layer)
                # Note: This basic version simulates etching layer1 using itself as a mask boundary
                etch_depth = param2 if param2 is not None else 100.0
                etch_speed = 10.0 # Example speed
                duration = etch_depth / etch_speed if etch_speed > 0 else 0
                self.run_level_set_simulation(
                    target_layer_name=layer1, # Simulating change *on* this layer's shapes
                    process_type="etch",
                    duration=duration,
                    speed=etch_speed,
                    grid_resolution=self.project.grid_pitch / 10.0 # Example resolution
                )
            else:
                 QMessageBox.warning(self, "Input Error", "Etching requires both a Mask and Substrate layer.")
                 return # Return here, don't save/redraw

        elif step == "Doping":
            active_cell = self.project.cells.get(self.active_cell_name)
            if not active_cell: return
            shapes_to_process = []
            if self.scene.selectedItems():
                 shapes_to_process = [item.data_obj for item in self.scene.selectedItems()
                                      if hasattr(item, 'data_obj') and item.data_obj.layer == layer1]
            else: # If nothing selected, process all shapes on the target layer in the current cell
                 shapes_to_process = [p for p in active_cell.polygons if p.layer == layer1]
                 shapes_to_process.extend([e for e in active_cell.ellipses if e.layer == layer1])
            dose = param1 if param1 is not None else 1e13
            self._apply_doping(shapes_to_process, layer1, dose, True)

        elif step == "Oxidation":
             self.placeholder_function(f"Oxidation: {ptype} on {layer1}, Thickness={param1}")
        else:
             self.placeholder_function(f"Process: {step} - {ptype}, L1={layer1}, L2={layer2}, P1={param1}, P2={param2}")

        # --- Common Actions ---
        # Save state and redraw are now typically handled within the specific apply/simulate functions
        # self._save_state() # Removed - happens in sub-functions
        # self._redraw_scene() # Removed - happens in sub-functions
        self.statusBar().showMessage(f"Applied {step} - {ptype}.") # Show generic message


    def _apply_deposition(self, layer_name, thickness, visual_effect):
        """Simulates deposition by adding a uniform layer."""
        target_layer = self.project.layer_by_name.get(layer_name)
        if not target_layer or thickness is None:
            return

        # Store process history as layer metadata
        if not hasattr(target_layer, 'process_history'):
            target_layer.process_history = []

        target_layer.process_history.append({
            'step': 'Deposition',
            'thickness': thickness
        })

        current = getattr(target_layer, 'thickness_2_5d', 10.0) # Ensure default is float
        target_layer.thickness_2_5d = max(0.0, current + thickness) # Ensure float arithmetic


        if visual_effect:
            # Slightly lighten the layer color to indicate deposition
            r, g, b = target_layer.color
            target_layer.color = (
                min(255, int(r * 1.05)), # Less aggressive change
                min(255, int(g * 1.05)),
                min(255, int(b * 1.05))
            )
        
        self._save_state() # Save state after modification
        self._redraw_scene() # Redraw needed for color change
        QMessageBox.information(self, "Deposition Complete",
            f"Added {thickness:.1f} units uniform layer on '{layer_name}'.\n"
            f"This affects 2.5D visualization thickness.")

    # UPDATED to set fill_pattern
    def _apply_etching(self, mask_layer_name, substrate_layer_name, etch_depth):
        """Simulates an etch by darkening the mask, setting its pattern, and recording an etch instruction."""
        if mask_layer_name == substrate_layer_name:
            QMessageBox.warning(self, "Logic Error", "Mask and Substrate layer cannot be the same.")
            return
        if etch_depth is None:
             QMessageBox.warning(self, "Input Error", "Etch depth parameter is missing.")
             return


        active_cell = self.project.cells.get(self.active_cell_name)
        mask_layer = self.project.layer_by_name.get(mask_layer_name)
        if not active_cell or not mask_layer:
            return

        # 1. Darken the mask layer for a simple 2D visual cue
        r, g, b = mask_layer.color
        mask_layer.color = (max(0, r - 30), max(0, g - 30), max(0, b - 30)) # Less aggressive

        # 2. SET THE FILL PATTERN to make it "special"
        mask_layer.fill_pattern = "hatch" 

        # 3. Record the etch instruction with depth in the cell's data
        etch_instruction = (mask_layer_name, substrate_layer_name, etch_depth)
        # Avoid duplicate instructions
        if not any(e == etch_instruction for e in active_cell.etch_pairs):
            active_cell.etch_pairs.append(etch_instruction)
        else:
            print(f"Skipping duplicate etch instruction.")


        # 4. Redraw the 2D scene and inform the user
        self._save_state() # Save state after modification
        self._redraw_scene() # Redraw needed for style change
        QMessageBox.information(self, "Etch Step Recorded",
            f"Mask '{mask_layer_name}' will now etch '{substrate_layer_name}' by a depth of {etch_depth} units in the 2.5D view.\nMask layer style updated.")

    # UPDATED to set fill_pattern
    def _apply_doping(self, shapes_to_process, layer_name, dose, visual_effect):
        """Simulates doping by moving shapes to a new 'doped' layer and annotating them."""
        if not shapes_to_process:
            return

        # Define the new doped layer
        original_layer = self.project.layer_by_name.get(layer_name)
        if not original_layer:
             QMessageBox.warning(self, "Error", f"Original layer '{layer_name}' not found for doping.")
             return
        doped_layer_name = f"{layer_name}_doped"
        doped_layer = self.project.layer_by_name.get(doped_layer_name)

        # Create the doped layer if it doesn't exist
        if not doped_layer:
            r, g, b = original_layer.color
            # Create a visually distinct color for the doped version
            new_color = (min(255, r + 50), max(0, g - 50), min(255, b + 50))
            # Set fill_pattern on creation
            doped_layer = Layer(name=doped_layer_name, color=new_color, fill_pattern="cross") # <-- MODIFIED
            self.project.layers.append(doped_layer)
            self.project.refresh_layer_map()
            self._refresh_layer_list() # Update the layer list widget

        # Move shapes to the new layer and annotate their names
        changed = False
        for shape in shapes_to_process:
            if shape.layer != doped_layer_name:
                 shape.layer = doped_layer_name
                 changed = True
            if hasattr(shape, 'name'):
                current_name = shape.name or ""
                new_name = f"{current_name} (Dose: {dose:.1E})".strip()
                if shape.name != new_name:
                    shape.name = new_name
                    changed = True # Name change also counts

        if changed:
            self._save_state() # Save state after modification
            self._redraw_scene() # Redraw needed for layer change
            QMessageBox.information(self, "Doping Complete",
                f"Applied doping to {len(shapes_to_process)} shape(s).\n"
                f"Shapes moved to new layer: '{doped_layer_name}'.")
        else:
             self.statusBar().showMessage("Selected shapes already doped with this dose.")


    # --- LEVEL SET SOLVER (Basic Implementation) ---
    def run_level_set_simulation(self, target_layer_name, process_type="etch", duration=10.0, speed=1.0, grid_resolution=1.0):
        """
        Performs a basic Level Set simulation on the target layer.
        process_type: 'etch' (positive speed) or 'deposit' (negative speed)
        duration: Total simulation time.
        speed: Rate of surface movement (positive for etch, negative for deposit).
        grid_resolution: Size of one grid cell in project units.
        """
        if not _has_level_set_deps: # Check if libraries loaded
             QMessageBox.warning(self, "Missing Libraries", "NumPy, SciPy, and Scikit-Image are required for Level Set simulation.")
             return
        if not self.project: return

        active_cell = self.project.cells.get(self.active_cell_name)
        if not active_cell: return

        target_polys_data = [p.points for p in active_cell.polygons if p.layer == target_layer_name]
        # TODO: Convert ellipses and flatten references for a complete simulation domain

        if not target_polys_data:
            self.statusBar().showMessage(f"No shapes found on layer '{target_layer_name}' for Level Set.")
            return

        # --- 1. Get Bounding Box ---
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        for points in target_polys_data:
            if not points: continue
            try:
                xs, ys = zip(*points)
                min_x, min_y = min(min_x, min(xs)), min(min_y, min(ys))
                max_x, max_y = max(max_x, max(xs)), max(max_y, max(ys))
            except ValueError: continue # Skip if points list was empty

        if not all(math.isfinite(v) for v in [min_x, min_y, max_x, max_y]):
             QMessageBox.warning(self, "Simulation Error", "Cannot determine bounds for simulation.")
             return

        margin = 5 * grid_resolution
        min_x -= margin; min_y -= margin; max_x += margin; max_y += margin

        # --- 2. Create Grid and Initial Phi ---
        grid_width = int(np.ceil((max_x - min_x) / grid_resolution))
        grid_height = int(np.ceil((max_y - min_y) / grid_resolution))
        if grid_width <= 1 or grid_height <= 1:
             QMessageBox.warning(self, "Simulation Error", "Simulation grid size too small. Check shapes or resolution.")
             return

        x_coords = np.linspace(min_x, max_x, grid_width)
        y_coords = np.linspace(min_y, max_y, grid_height)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T # Shape (N, 2)

        inside_mask = np.zeros(grid_points.shape[0], dtype=bool)
        if Path: # Check if Path is available
            for points in target_polys_data:
                 if len(points) >= 3: # Path requires at least 3 points
                     path = Path(points)
                     inside_mask |= path.contains_points(grid_points)
        else:
             QMessageBox.warning(self, "Missing Library", "Matplotlib Path needed for shape containment check.")
             return

        inside_mask = inside_mask.reshape(grid_height, grid_width)

        try:
             # Calculate signed distance function (SDF) phi
             phi = distance_transform_edt(inside_mask) - distance_transform_edt(~inside_mask)
        except Exception as e:
             QMessageBox.critical(self, "SDF Error", f"Failed to compute initial distance field: {e}")
             return

        # --- 3. Simple Simulation Loop ---
        safe_speed = max(abs(speed), 1e-9) # Avoid division by zero
        dt = 0.4 * grid_resolution / safe_speed # CFL condition estimate (0.4 is safer than 0.1)
        if dt <= 0:
             QMessageBox.warning(self, "Simulation Error", "Calculated timestep is zero or negative.")
             return
        num_steps = int(duration / dt) if duration > 0 else 0
        if num_steps <= 0:
            QMessageBox.information(self, "Simulation Info", "Duration or speed results in zero simulation steps.")
            return


        current_phi = phi.copy()
        F = speed if process_type == "etch" else -abs(speed)

        print(f"Starting Level Set: {num_steps} steps, dt={dt:.4f}, F={F:.2f}")
        QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor
        try:
            for step in range(num_steps):
                phi_x_plus = np.roll(current_phi, -1, axis=1) - current_phi
                phi_x_minus = current_phi - np.roll(current_phi, 1, axis=1)
                phi_y_plus = np.roll(current_phi, -1, axis=0) - current_phi
                phi_y_minus = current_phi - np.roll(current_phi, 1, axis=0)

                # Upwind gradient magnitude squared calculation
                grad_phi_sq_term_x = np.zeros_like(current_phi)
                grad_phi_sq_term_y = np.zeros_like(current_phi)

                if F > 0: # Etching
                    grad_phi_sq_term_x = np.maximum(phi_x_minus / grid_resolution, 0)**2 + np.minimum(phi_x_plus / grid_resolution, 0)**2
                    grad_phi_sq_term_y = np.maximum(phi_y_minus / grid_resolution, 0)**2 + np.minimum(phi_y_plus / grid_resolution, 0)**2
                else: # Deposition
                    grad_phi_sq_term_x = np.minimum(phi_x_minus / grid_resolution, 0)**2 + np.maximum(phi_x_plus / grid_resolution, 0)**2
                    grad_phi_sq_term_y = np.minimum(phi_y_minus / grid_resolution, 0)**2 + np.maximum(phi_y_plus / grid_resolution, 0)**2

                grad_phi_mag = np.sqrt(grad_phi_sq_term_x + grad_phi_sq_term_y)

                # Update phi: phi_t = -F * |grad(phi)|
                current_phi -= dt * F * grad_phi_mag

                # Basic progress update
                if step % (max(1, num_steps // 10)) == 0:
                    print(f" Step {step}/{num_steps}")
                    QApplication.processEvents() # Keep UI responsive

            print("Level Set Simulation Finished.")
        except Exception as e:
            QMessageBox.critical(self, "Simulation Runtime Error", f"Error during level set update: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()


        # --- 4. Extract Contours ---
        try:
            # Note: find_contours assumes y, x order (row, col)
            contours = find_contours(current_phi, level=0.0)
        except Exception as e:
            QMessageBox.critical(self, "Contour Error", f"Failed to extract contours: {e}")
            return

        # --- 5. Add Results to Project ---
        result_polygons_data = []
        for contour in contours:
            # Convert contour indices (row, col) back to project coordinates (x, y)
            points_xy = []
            for row, col in contour:
                # Interpolation for slightly smoother coords
                x = min_x + col * grid_resolution
                y = min_y + row * grid_resolution
                points_xy.append((x, y))

            if len(points_xy) >= 3:
                 result_polygons_data.append(points_xy)

        if result_polygons_data:
            result_layer_name = f"{target_layer_name}_sim_{process_type}"
            result_layer = self.project.layer_by_name.get(result_layer_name)
            if not result_layer:
                original_color = self.project.layer_by_name[target_layer_name].color
                # Make result color slightly different
                new_r = min(255, max(0, original_color[0] + (30 if process_type=='deposit' else -30)))
                new_g = min(255, max(0, original_color[1] + (30 if process_type=='deposit' else -30)))
                new_b = min(255, max(0, original_color[2] + (30 if process_type=='deposit' else -30)))
                result_layer = Layer(name=result_layer_name, color=(new_r, new_g, new_b), fill_pattern="dots")
                self.project.layers.append(result_layer)
                self.project.refresh_layer_map()
                self._refresh_layer_list()

            num_added = 0
            for points in result_polygons_data:
                active_cell.polygons.append(Poly(layer=result_layer_name, points=points))
                num_added += 1

            if num_added > 0:
                self._save_state()
                self._redraw_scene()
                self.statusBar().showMessage(f"Level Set result ({num_added} polys) added to '{result_layer_name}'.")
            else:
                 self.statusBar().showMessage("Level Set simulation complete, but no valid polygons generated.")
        else:
            self.statusBar().showMessage("Level Set simulation did not produce any contours.")


    # --- Placeholder functions for new UI buttons ---
    def placeholder_function(self, message="Not implemented yet."):
         QMessageBox.information(self, "Placeholder", message)

    def run_recipe_loop(self): self.placeholder_function("Run Recipe Loop (DRIE/Bosch)")
    def show_virtual_cross_section(self): self.placeholder_function("Show Virtual Cross-Section")
    def reset_process_simulation(self):
        # Clear etch pairs and potentially reset layer thicknesses/colors/patterns
        if not self.project: return
        active_cell = self.project.cells.get(self.active_cell_name)
        if active_cell:
            active_cell.etch_pairs.clear()
        # Reset layer appearances (optional - might want to keep thickness changes)
        # for layer in self.project.layers:
        #    layer.fill_pattern = 'solid'
        #    # Reset color? Reset thickness?
        self._save_state()
        self._redraw_scene()
        QMessageBox.information(self, "Process Reset", "Simulated 2.5D etch effects have been cleared.")

    def run_locos(self): self.placeholder_function("Run Quick Recipe: Grow Oxide (LOCOS)")
    def run_lift_off(self): self.placeholder_function("Run Quick Recipe: Metal Lift-off")
    def run_sti_etch(self): self.placeholder_function("Run Quick Recipe: Trench Etch (STI)")
    def run_damascene(self): self.placeholder_function("Run Quick Recipe: Damascene Fill")

    def toggle_print_contour(self, checked): self.placeholder_function(f"Toggle Print Contour (AI): {'ON' if checked else 'OFF'}")
    def detect_hotspots(self): self.placeholder_function("Detect Hotspots (AI)")
    def run_contour_metrology(self): self.placeholder_function("Run Contour Metrology (AI)")


    # --- UI Building & Core Logic ---

    def _create_themed_icon(self, icon_name, size=32):
        # 1. Attempt to find the file
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
        icon_path = os.path.join(script_dir, 'icons', f'{icon_name}.svg')
        
        pixmap = QPixmap(icon_path)

        # 2. Check if load failed. If so, create a blank placeholder.
        if pixmap.isNull():
            # Create a transparent placeholder so Painter has a valid target
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.transparent)
            
            # Optional: Draw the first letter of the icon name as a fallback text
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.gray))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, icon_name[:1].upper())
            painter.end()
            return QIcon(pixmap)

        # 3. Apply Dark Mode theme logic ONLY if the pixmap loaded successfully
        is_dark = self.palette().color(QPalette.Window).value() < 128
        if is_dark:
            painter = QPainter(pixmap)
            # This line caused the crash previously if pixmap was null
            painter.setCompositionMode(QPainter.CompositionMode_SourceIn) 
            painter.fillRect(pixmap.rect(), Qt.white)
            painter.end()
            
        return QIcon(pixmap)

    def _build_ui(self):
        self.tool_buttons = {}
        self._create_shortcuts()
        self._build_tool_tabs() # This now builds all tabs including the new one
        self._build_cell_dock(); self._build_layer_dock()
        self.set_active_tool("select")

    def _create_shortcuts(self):
        shortcuts = [
            # File & Edit
            ("New", "Ctrl+N", self.new_doc),
            ("Open...", "Ctrl+O", self.open_file),
            ("Save", "Ctrl+S", self.save_json),
            ("Save As...", "Ctrl+Shift+S", self.save_json_as),
            ("Undo", "Ctrl+Z", self.undo),
            ("Redo", "Ctrl+Y", self.redo),
            ("Copy", "Ctrl+C", self.copy_selected_shapes),
            ("Paste", "Ctrl+V", self.paste_shapes),
            ("Delete", ("Del", "Backspace"), self.delete_selected_items),
            ("Select All", "Ctrl+A", self.select_all_items),
            ("Quit", "Ctrl+Q", self.close),
            # View
            ("Fit View", "F", self.fill_view),
            ("Zoom In", "Ctrl+=", self.zoom_in),
            ("Zoom Out", "Ctrl+-", self.zoom_out),
            # Tools
            ("Select Tool", "S", lambda: self.set_active_tool("select")),
            ("Move Tool", "M", lambda: self.set_active_tool("move")),
            ("Rectangle Tool", "R", lambda: self.set_active_tool("rect")),
            ("Polygon Tool", "P", lambda: self.set_active_tool("poly")),
            ("Circle Tool", "C", lambda: self.set_active_tool("circle")),
            ("Path Tool", "W", lambda: self.set_active_tool("path")),
        ]

        for text, shortcut_keys, slot in shortcuts:
            action = QAction(text, self)
            if isinstance(shortcut_keys, (list, tuple)):
                action.setShortcuts(shortcut_keys)
            else:
                action.setShortcut(shortcut_keys)
            action.setShortcutContext(Qt.ApplicationShortcut)
            action.triggered.connect(slot)
            self.addAction(action)

    # UPDATED: Builds all tabs now
    def _build_tool_tabs(self):
        dock = QDockWidget(self); dock.setTitleBarWidget(QWidget()); dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetMovable); self.tabs = QTabWidget(); dock.setWidget(self.tabs) # Store tabs ref

        # --- Main Tab ---
        main_tab = QWidget(); main_layout = QHBoxLayout(main_tab)
        main_layout.setAlignment(Qt.AlignLeft); main_layout.setContentsMargins(0, 5, 0, 5)

        export_button = QToolButton(); export_button.setText("Export"); export_button.setIcon(self._create_themed_icon("upload"))
        export_button.setIconSize(QSize(24, 24)); export_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        export_button.setPopupMode(QToolButton.MenuButtonPopup)
        export_menu = QMenu(self)
        if gdstk:
            gds = export_menu.addAction("Export GDS..."); gds.triggered.connect(lambda: self._write_gdstk(False))
            oas = export_menu.addAction("Export OAS..."); oas.triggered.connect(lambda: self._write_gdstk(True))
        svg = export_menu.addAction("Export SVG..."); svg.triggered.connect(self.export_svg)
        export_button.setMenu(export_menu)
        main_layout.addWidget(self._create_ribbon_group("Project", [
            self._create_action_button("New", "file-plus", self.new_doc),
            self._create_action_button("Open", "folder", self.open_file),
            self._create_action_button("Save", "save", self.save_json), export_button
        ]))
        main_layout.addWidget(self._create_ribbon_group("Selection", [
            self._create_tool_button("select", "Select", "mouse-pointer"), self._create_tool_button("move", "Move", "move")
        ]))
        main_layout.addWidget(self._create_ribbon_group("Clipboard", [
            self._create_action_button("Copy", "copy", self.copy_selected_shapes),
            self._create_action_button("Paste", "clipboard", lambda: self.paste_shapes(self.view.mapToScene(self.view.viewport().rect().center()) if hasattr(self, 'view') else QPointF(0,0))),
            self._create_action_button("Delete", "trash-2", self.delete_selected_items),
        ]))
        main_layout.addWidget(self._create_ribbon_group("History", [
            self._create_action_button("Undo", "corner-up-left", self.undo),
            self._create_action_button("Redo", "corner-up-right", self.redo)
        ]))
        grid_widget = QWidget(); grid_layout = QVBoxLayout(grid_widget)
        self.chk_snap = QCheckBox("Snap"); self.chk_snap.setChecked(True)
        self.spin_grid = QSpinBox(); self.spin_grid.setRange(1, 1000)
        self.spin_grid.valueChanged.connect(self.set_grid_pitch)
        grid_layout.addWidget(self.chk_snap); grid_size_layout = QHBoxLayout()
        grid_size_layout.addWidget(QLabel("Grid:")); grid_size_layout.addWidget(self.spin_grid)
        grid_layout.addLayout(grid_size_layout)
        main_layout.addWidget(self._create_ribbon_group("Grid", [grid_widget]))
        main_layout.addWidget(self._create_ribbon_group("Help", [
            self._create_action_button("User Guide", "help-circle", self.show_help_pdf)
        ]))
        main_layout.addStretch(); self.tabs.addTab(main_tab, "Main")

        # --- Draw Tab ---
        draw_tab = QWidget(); draw_layout = QHBoxLayout(draw_tab)
        draw_layout.setAlignment(Qt.AlignLeft); draw_layout.setContentsMargins(0, 5, 0, 5)
        draw_layout.addWidget(self._create_ribbon_group("Create", [
            self._create_tool_button("rect", "Rectangle", "square"),
            self._create_tool_button("circle", "Circle", "circle"),
            self._create_tool_button("poly", "Polygon", "pen-tool"),
            self._create_action_button("Finish Poly", "check-square", lambda: self.view.finish_polygon() if hasattr(self, 'view') else None),
            self._create_tool_button("path", "Path/Wire", "git-branch")
        ]))

        move_layer_btn = QToolButton(); move_layer_btn.setText("Move Layer"); move_layer_btn.setIcon(self._create_themed_icon("layers"))
        move_layer_btn.setIconSize(QSize(24, 24)); move_layer_btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        move_layer_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.move_layer_menu = QMenu(self)
        move_layer_btn.setMenu(self.move_layer_menu)

        arrange_btn = QToolButton()
        arrange_btn.setText("Arrange")
        arrange_btn.setIcon(self._create_themed_icon("bar-chart-2"))
        arrange_btn.setIconSize(QSize(24, 24))
        arrange_btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        arrange_btn.setPopupMode(QToolButton.MenuButtonPopup)

        arrange_menu = QMenu(self)
        bf_action = arrange_menu.addAction("Bring to Front")
        bf_action.triggered.connect(self.bring_selection_to_front)
        mf_action = arrange_menu.addAction("Move Forward")
        mf_action.triggered.connect(self.move_selection_forward)
        mb_action = arrange_menu.addAction("Move Backward")
        mb_action.triggered.connect(self.move_selection_backward)
        sb_action = arrange_menu.addAction("Send to Back")
        sb_action.triggered.connect(self.send_selection_to_back)
        arrange_btn.setMenu(arrange_menu)

        draw_layout.addWidget(self._create_ribbon_group("Modify", [
            self._create_action_button("Rename", "tag", self.rename_selected_shape),
            self._create_action_button("Re-snap", "grid", self.resnap_all_items_to_grid),
            self._create_action_button("Fillet", "git-commit", self.fillet_selected_poly),
            move_layer_btn, arrange_btn
        ]))
        draw_layout.addWidget(self._create_ribbon_group("Transform", [
            self._create_action_button("Rotate", "rotate-cw", self.rotate_selection),
            self._create_action_button("Scale", "zoom-in", self.scale_selection),
            self._create_action_button("Flip", "triangle", self.flip_selection)
        ]))
        draw_layout.addWidget(self._create_ribbon_group("Boolean", [
            self._create_action_button("Union", "plus-square", lambda: self._run_boolean_op('or')),
            self._create_action_button("Subtract", "minus-square", lambda: self._run_boolean_op('not')),
            self._create_action_button("Intersect", "crop", lambda: self._run_boolean_op('and'))
        ]))
        # REMOVED Process button from here
        draw_layout.addStretch(); self.tabs.addTab(draw_tab, "Draw")

        # --- View Tab ---
        view_tab = QWidget(); view_layout = QHBoxLayout(view_tab)
        view_layout.setAlignment(Qt.AlignLeft); view_layout.setContentsMargins(0, 5, 0, 5)
        toggle_rulers_btn = self._create_action_button("Rulers", "table", lambda checked: self.toggle_rulers(checked))
        toggle_rulers_btn.setCheckable(True); toggle_rulers_btn.setChecked(True)
        view_layout.addWidget(self._create_ribbon_group("Display", [
            self._create_action_button("Fill", "maximize", self.fill_view),
            self._create_action_button("2.5D View", "box", self.show_2_5d_view), # RENAMED
            toggle_rulers_btn, self._create_tool_button("measure", "Measure", "compass")
        ]))
        view_layout.addWidget(self._create_ribbon_group("Zoom", [
            self._create_action_button("Zoom In", "zoom-in", self.zoom_in), self._create_action_button("Zoom Out", "zoom-out", self.zoom_out)
        ]))
        view_layout.addWidget(self._create_ribbon_group("Layer Visibility", [
            self._create_action_button("Show All", "eye", self.show_all_layers), self._create_action_button("Hide All", "eye-off", self.hide_all_layers)
        ]))
        view_layout.addStretch(); self.tabs.addTab(view_tab, "View")

        # --- Process Tab (NEW) ---
        process_tab = QWidget(); process_layout = QHBoxLayout(process_tab)
        process_layout.setAlignment(Qt.AlignLeft); process_layout.setContentsMargins(0, 5, 0, 5)

        process_layout.addWidget(self._create_ribbon_group("Process Simulation", [
            self._create_action_button("Add Step...", "sliders", self.run_process_step), # Opens enhanced dialog
            self._create_action_button("Loop...", "repeat", self.run_recipe_loop),
            self._create_action_button("Show 2.5D", "box", self.show_2_5d_view),
            self._create_action_button("Show 3D", "box", self.show_3d_view),
            self._create_action_button("X-Section", "scissors", self.show_virtual_cross_section),
            self._create_action_button("Reset", "refresh-cw", self.reset_process_simulation),
        ]))

        process_layout.addWidget(self._create_ribbon_group("Quick Recipes", [
             self._create_action_button("LOCOS", "thermometer", self.run_locos),
             self._create_action_button("Lift-off", "chevrons-up-down", self.run_lift_off),
             self._create_action_button("STI Etch", "chevrons-down", self.run_sti_etch),
             self._create_action_button("Damascene", "droplet", self.run_damascene),
        ]))

        process_layout.addStretch(); self.tabs.addTab(process_tab, "Process") # NEW TAB

        # --- Lithography Tab (NEW) ---
        litho_tab = QWidget(); litho_layout = QHBoxLayout(litho_tab)
        litho_layout.setAlignment(Qt.AlignLeft); litho_layout.setContentsMargins(0, 5, 0, 5)

        contour_toggle_btn = self._create_action_button("Contour", "aperture", self.toggle_print_contour)
        contour_toggle_btn.setCheckable(True) # Make it a toggle button
        litho_layout.addWidget(self._create_ribbon_group("Lithography (AI)", [
            contour_toggle_btn,
            self._create_action_button("Hotspots", "alert-triangle", self.detect_hotspots),
            self._create_action_button("Measure CD", "ruler", self.run_contour_metrology),
        ]))

        litho_layout.addStretch(); self.tabs.addTab(litho_tab, "Lithography") # NEW TAB

        # Add the dock widget containing all tabs
        self.addDockWidget(Qt.TopDockWidgetArea, dock)


    def _create_tool_button(self, name, text, icon):
        btn = QToolButton(); btn.setText(text); btn.setIcon(self._create_themed_icon(icon))
        btn.setIconSize(QSize(24, 24)); btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setCheckable(True); btn.clicked.connect(lambda: self.set_active_tool(name))
        self.tool_buttons[name] = btn
        return btn

    def _create_action_button(self, text, icon, slot):
        btn = QToolButton(); btn.setText(text); btn.setIcon(self._create_themed_icon(icon))
        btn.setIconSize(QSize(24, 24)); btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.clicked.connect(slot)
        return btn

    def _create_ribbon_group(self, title, widgets):
        container = QWidget(); layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        btn_widget = QWidget(); btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(10, 5, 10, 5); btn_layout.setSpacing(5)
        panel = QWidget(); panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0,0,0,0); panel_layout.setSpacing(2); panel_layout.setAlignment(Qt.AlignBottom)
        for w in widgets: btn_layout.addWidget(w)
        title_label = QLabel(title); title_label.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(btn_widget); panel_layout.addWidget(title_label)
        layout.addWidget(panel); separator = QFrame()
        separator.setFrameShape(QFrame.VLine); separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        return container

    def set_active_tool(self, tool_name):
        if hasattr(self, 'view') and self.view: self.view.set_mode(tool_name)
        for name, btn in self.tool_buttons.items(): btn.setChecked(name == tool_name)
        # Ensure correct tab is shown when a tool from a specific tab is activated
        if hasattr(self, 'tabs'): # Check if tabs are built yet
            if tool_name in ["rect", "circle", "poly", "path"]:
                self.tabs.setCurrentIndex(1) # Draw tab
            elif tool_name in ["measure"]:
                 self.tabs.setCurrentIndex(2) # View tab
            elif tool_name in ["select", "move"]:
                 self.tabs.setCurrentIndex(0) # Main tab

    def _build_cell_dock(self):
        dock = QDockWidget("Cells", self)
        self.list_cells = CellListWidget()
        self.list_cells.setDragEnabled(True)
        self.list_cells.itemDoubleClicked.connect(self._on_cell_double_clicked)
        widget, layout = QWidget(), QVBoxLayout()
        layout.addWidget(QLabel("Double-click to edit, Drag to instantiate"));
        layout.addWidget(self.list_cells)
        btns = QHBoxLayout()
        for label, func in [("Add", self.add_cell_dialog), ("Rename", self.rename_cell_dialog), ("Delete", self.delete_cell_dialog)]:
            btn = QPushButton(label); btn.clicked.connect(func); btns.addWidget(btn)
        layout.addLayout(btns); widget.setLayout(layout); dock.setWidget(widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _build_layer_dock(self):
        dock = QDockWidget("Layers", self); self.list_layers = QListWidget()
        self.list_layers.itemChanged.connect(self._update_layer_visibility)
        self.list_layers.itemSelectionChanged.connect(self._on_layer_selected)
        self.list_layers.itemDoubleClicked.connect(self.change_layer_color_dialog)
        widget = QWidget(); layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Toggle visibility, double-click color")); layout.addWidget(self.list_layers)
        btns1 = QHBoxLayout()
        layer_actions = [("Move to Top (drawn last)", "chevrons-up", self.move_layer_to_top), ("Move Up", "chevron-up", self.move_layer_forward), ("Move Down", "chevron-down", self.move_layer_backward), ("Move to Bottom (drawn first)", "chevrons-down", self.move_layer_to_bottom)]
        for tooltip, icon_name, func in layer_actions:
            btn = QToolButton(); btn.setIcon(self._create_themed_icon(icon_name)); btn.setToolTip(tooltip)
            btn.clicked.connect(func); btns1.addWidget(btn)
        layout.addLayout(btns1); separator = QFrame()
        separator.setFrameShape(QFrame.HLine); separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator); btns2 = QHBoxLayout()
        for label, func in [("Add", self.add_layer_dialog), ("Rename", self.rename_layer_dialog), ("Delete", self.delete_layer_dialog)]:
            btn = QPushButton(label); btn.clicked.connect(func); btns2.addWidget(btn)
        layout.addLayout(btns2); dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def update_ui_from_project(self):
        if not self.project: return
        if self.project.top not in self.project.cells:
            self.project.top = list(self.project.cells.keys())[0] if self.project.cells else None
        if not self.active_cell_name or self.active_cell_name not in self.project.cells:
            self.active_cell_name = self.project.top
        # Ensure spin_grid exists before setting value
        if hasattr(self, 'spin_grid'):
            self.spin_grid.setValue(self.project.grid_pitch)
        self._refresh_cell_list(); self._refresh_layer_list(); self._redraw_scene()

    def _refresh_cell_list(self):
        self.list_cells.blockSignals(True)
        current = self.active_cell_name; self.list_cells.clear()
        for name in sorted(self.project.cells.keys()):
            item = QListWidgetItem(name); self.list_cells.addItem(item)
            if name == current: item.setSelected(True)
        self.list_cells.blockSignals(False)

    def _refresh_layer_list(self):
        self.list_layers.blockSignals(True)
        current = self.active_layer_name; self.list_layers.clear()
        # Ensure move_layer_menu exists before clearing
        if hasattr(self, 'move_layer_menu'):
            self.move_layer_menu.clear()
        for layer in self.project.layers:
            item = QListWidgetItem(layer.name); item.setIcon(self._create_color_icon(QColor(*layer.color)))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable); item.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
            self.list_layers.addItem(item)
            if layer.name == current: self.list_layers.setCurrentItem(item)
            if hasattr(self, 'move_layer_menu'):
                action = self.move_layer_menu.addAction(layer.name)
                action.triggered.connect(lambda checked=False, l_name=layer.name: self.move_selection_to_layer(l_name))
        self.list_layers.blockSignals(False)

    def _create_color_icon(self, color, size=16):
        pixmap = QPixmap(size, size); pixmap.fill(Qt.transparent)
        p = QPainter(pixmap); p.setBrush(QBrush(color)); p.setPen(Qt.black)
        p.drawRect(0, 0, size - 1, size - 1); p.end()
        return QIcon(pixmap)

    def _on_cell_double_clicked(self, item):
        if item.text() != self.active_cell_name:
            self.active_cell_name = item.text(); self._redraw_scene()
    def _on_layer_selected(self):
        if item := self.list_layers.currentItem(): self.active_layer_name = item.text()
    def _update_layer_visibility(self, item):
        if (layer := self.project.layer_by_name.get(item.text())):
            layer.visible = item.checkState() == Qt.Checked
            self._redraw_scene(); self._save_state()

    def active_layer(self):
        return self.project.layer_by_name.get(self.active_layer_name) if self.project else None

    def _save_state(self):
        if not self.project: return
        self._redo_stack.clear()
        # Save project state *before* potential modification
        current_state = copy.deepcopy(self.project)
        # Avoid saving identical consecutive states
        if not self._undo_stack or self._undo_stack[-1] != current_state:
             self._undo_stack.append(current_state)
             if len(self._undo_stack) > 50: self._undo_stack.pop(0)
             if len(self._undo_stack) > 1: self._mark_dirty()


    def undo(self):
        if len(self._undo_stack) > 1:
            self._redo_stack.append(self._undo_stack.pop())
            self.project = copy.deepcopy(self._undo_stack[-1])
            self.update_ui_from_project(); self._mark_dirty() # Should mark as potentially dirty after undo
            if len(self._undo_stack) == 1: # If only initial state remains, mark as clean
                self._is_dirty = False
                self._update_window_title()

    def redo(self):
        if self._redo_stack:
            # Need deepcopy here too, otherwise undo/redo share the same object
            state_to_restore = copy.deepcopy(self._redo_stack.pop())
            self._undo_stack.append(state_to_restore) # Add the redone state back to undo
            self.project = state_to_restore # This assignment is okay now
            self.update_ui_from_project(); self._mark_dirty()


    def add_rect_to_active_cell(self, rect, layer):
        pts = [(rect.left(), rect.top()), (rect.right(), rect.top()), (rect.right(), rect.bottom()), (rect.left(), rect.bottom())]
        self.project.cells[self.active_cell_name].polygons.append(Poly(layer.name, pts))
        self._redraw_scene(); self._save_state()
    def add_circle_to_active_cell(self, rect, layer):
        self.project.cells[self.active_cell_name].ellipses.append(Ellipse(layer.name, (rect.x(), rect.y(), rect.width(), rect.height())))
        self._redraw_scene(); self._save_state()
    def add_poly_to_active_cell(self, poly, layer):
        self.project.cells[self.active_cell_name].polygons.append(Poly(layer.name, [(p.x(), p.y()) for p in poly]))
        self._redraw_scene(); self._save_state()
    def add_path_to_active_cell(self, points: List[QPointF], layer):
        if not gdstk:
            QMessageBox.warning(self, "Feature Disabled", "The Path/Wire tool requires the 'gdstk' library to be installed.")
            return

        sensible_width = self.get_sensible_default_value(divisor=100.0)

        width, ok = QInputDialog.getDouble(self, "Path Width", "Enter path width:", sensible_width, 0.001, 10000.0, 3)

        if not ok: return

        try:
            point_list = [(p.x(), p.y()) for p in points]
            path = gdstk.FlexPath(point_list, width)
            new_polygons = path.to_polygons() # This can return multiple polygons

            active_cell = self.project.cells[self.active_cell_name]
            for poly in new_polygons:
                active_cell.polygons.append(Poly(layer.name, poly.points.tolist()))

            self._redraw_scene(); self._save_state()
        except Exception as e:
            QMessageBox.critical(self, "Path Creation Failed", f"Could not create the path: {e}")

    def add_reference_to_active_cell(self, cell_name, pos):
        if cell_name != self.active_cell_name:

            bounds = self.get_cell_bounds(cell_name)

            if not bounds.isNull():
                center_offset_x = bounds.center().x()
                center_offset_y = bounds.center().y()

                new_origin_x = pos.x() - center_offset_x
                new_origin_y = pos.y() - center_offset_y
            else:
                new_origin_x, new_origin_y = pos.x(), pos.y()

            new_ref = Ref(cell_name, (new_origin_x, new_origin_y))
            self.project.cells[self.active_cell_name].references.append(new_ref)
            self._save_state()
            self._redraw_scene()

            for item in self.scene.items():
                if isinstance(item, RefItem) and item.ref == new_ref:
                    self.scene.clearSelection()
                    item.setSelected(True)
                    break

            self.statusBar().showMessage(f"Instantiated cell '{cell_name}' at ({pos.x():.2f}, {-pos.y():.2f})")

    def instantiate_cell_on_shape(self, cell_name, target):
        if isinstance(target, Poly):
            xs, ys = [p[0] for p in target.points], [p[1] for p in target.points]
            placeholder = QRectF(QPointF(min(xs), min(ys)), QPointF(max(xs), max(ys)))
        elif isinstance(target, Ellipse): placeholder = QRectF(*target.rect)
        else: return

        bounds = self.get_cell_bounds(cell_name); mag = 1.0

        if not bounds.isNull() and bounds.width() > 0 and bounds.height() > 0:
            mag_x = placeholder.width() / bounds.width()
            mag_y = placeholder.height() / bounds.height()
            mag = min(mag_x, mag_y)
        else:
            mag=1.0

        origin_x = placeholder.center().x() - (bounds.center().x() * mag)
        origin_y = placeholder.center().y() - (bounds.center().y() * mag)

        active_cell = self.project.cells[self.active_cell_name]

        target_name = target.name or type(target).__name__

        if isinstance(target, Poly): active_cell.polygons = [p for p in active_cell.polygons if p.uuid != target.uuid]
        elif isinstance(target, Ellipse): active_cell.ellipses = [e for e in active_cell.ellipses if e.uuid != target.uuid]

        new_ref = Ref(cell=cell_name, origin=(origin_x, origin_y), magnification=mag)
        active_cell.references.append(new_ref)

        self._save_state(); self._redraw_scene()

        for item in self.scene.items():
            if isinstance(item, RefItem) and item.ref == new_ref:
                self.scene.clearSelection()
                item.setSelected(True)
                break

        self.statusBar().showMessage(f"Instantiated cell '{cell_name}' on '{target_name}' and deleted placeholder.")

    def update_data_from_item_move(self, item):
        # Snap logic now part of itemChange, just save the final state
        pos = item.pos()
        if isinstance(item, RefItem):
            item.ref.origin = (pos.x(), pos.y())
        elif isinstance(item, CircleItem):
            # Recalculate rect based on item's new scene bounding rect
            rect = item.sceneBoundingRect()
            item.data_obj.rect = (rect.x(), rect.y(), rect.width(), rect.height())
        elif isinstance(item, (PolyItem, RectItem)):
             poly = item.mapToScene(item.polygon() if isinstance(item, PolyItem) else item.rect())
             item.data_obj.points = [(p.x(), p.y()) for p in poly]

        self._save_state()
        # No redraw needed, item is already at new position visually
        
    def update_data_from_item_edit(self, item):
        if isinstance(item, RectItem):
            r = item.rect()
            item.data_obj.points = [(r.left(), r.top()), (r.right(), r.top()), (r.right(), r.bottom()), (r.left(), r.bottom())]
        elif isinstance(item, PolyItem):
            item.data_obj.points = [(p.x(), p.y()) for p in item.polygon()]
        elif isinstance(item, CircleItem):
            r = item.rect()
            item.data_obj.rect = (r.x(), r.y(), r.width(), r.height())
        self._save_state()

    def delete_selected_items(self):
        if not (self.project and (cell := self.project.cells.get(self.active_cell_name)) and (selected := self.scene.selectedItems())): return

        to_delete_data = {item.data_obj for item in selected if hasattr(item, 'data_obj')}
        to_delete_refs = {item.ref for item in selected if isinstance(item, RefItem)}

        if to_delete_data or to_delete_refs:
            self.scene.clearSelection() # Clear selection before modifying list
            cell.polygons = [p for p in cell.polygons if p not in to_delete_data]
            cell.ellipses = [e for e in cell.ellipses if e not in to_delete_data]
            cell.references = [r for r in cell.references if r not in to_delete_refs]
            self._redraw_scene(); self._save_state()


    def fillet_selected_poly(self):
        if not gdstk: QMessageBox.warning(self, "Feature Disabled", "Please install 'gdstk'."); return
        selected = [it for it in self.scene.selectedItems() if isinstance(it, PolyItem)]
        if not selected: self.statusBar().showMessage("Select one or more polygons to fillet."); return

        sensible_radius = self.get_sensible_default_value(divisor=50.0) # A fillet is usually larger

        dlg = FilletDialog(self, default_radius=sensible_radius)

        if dlg.exec_() == QDialog.Accepted:
            radius = dlg.get_radius()
            for item in selected:
                try:
                    gds_poly = gdstk.Polygon([(p.x(), p.y()) for p in item.polygon()]); gds_poly.fillet(radius)
                    item.data_obj.points = gds_poly.points.tolist()
                except Exception as e: QMessageBox.warning(self, "Fillet Error", f"Could not fillet polygon: {e}")
            self._save_state(); self._redraw_scene()

    def fill_view(self):
        if not (hasattr(self, 'scene') and self.scene.items()): return # Check scene exists
        rect = self.scene.itemsBoundingRect()
        if not rect.isNull():
            margin = max(rect.width(), rect.height()) * 0.05
            if margin == 0: margin = 10 # Add default if empty or single point
            self.view.fitInView(rect.adjusted(-margin, -margin, margin, margin), Qt.KeepAspectRatio)
            if hasattr(self, 'view'): # Check view exists before emitting
                self.view.zoomChanged.emit()

    def show_3d_view(self):
        if not self.project: return
        if not _has_3d_deps:
            QMessageBox.warning(self, "Missing Libraries", 
                "Install 'pyqtgraph', 'pyopengl', and 'scikit-image' to use the 3D Voxel View.")
            return
            
        dlg = ThreeDViewDialog(self.project, self)
        dlg.exec_()

    # RENAMED from show_3d_view
    def show_2_5d_view(self):
        if self.project: TwoPointFiveDViewDialog(self.project, self).exec_()

    def export_svg(self):
        if not self.project:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", "", "SVG Files (*.svg)")
        if not path:
            return

        # 1. Get the exact bounding rectangle of all items
        items_rect = self.scene.itemsBoundingRect()

        if not items_rect.isValid() or items_rect.isEmpty():
            QMessageBox.warning(self, "Export Error", "The current cell is empty. Nothing to export.")
            return

        # 2. Add a small margin for clarity
        margin = max(items_rect.width(), items_rect.height()) * 0.05
        if margin == 0: margin = 10 # Add a default margin for very small designs
        export_rect = items_rect.adjusted(-margin, -margin, margin, margin)

        # 3. Set up the SVG generator
        gen = QSvgGenerator()
        gen.setFileName(path)
        gen.setSize(export_rect.size().toSize())
        gen.setViewBox(export_rect)

        # 4. Render the scene
        painter = QPainter(gen)
        
        # We need to render the background *if* it's not white, otherwise SVG is transparent
        # bg_color = self.palette().color(QPalette.Window)
        # if bg_color != Qt.white:
        #     painter.fillRect(export_rect, bg_color)
            
        # CRITICAL FIX: Explicitly tell render() which SOURCE rectangle from the scene to draw.
        # Because the viewBox is already set, we don't need a target rectangle.
        self.scene.render(painter, source=export_rect)

        painter.end()

        self.statusBar().showMessage(f"Successfully exported SVG to {path}")

    def set_grid_pitch(self, value):
        if self.project: self.project.grid_pitch = value
        if hasattr(self, 'view'): self.view.viewport().update() # Redraw background grid
        self._mark_dirty()


    def resnap_all_items_to_grid(self):
        if not self.project: return
        reply = QMessageBox.question(self, 'Re-snap Cell', "This will snap all shapes in the current cell to the grid. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No or self.project.grid_pitch <= 0: return
        if not (active_cell := self.project.cells.get(self.active_cell_name)): return
        def snap(v): return round(v / self.project.grid_pitch) * self.project.grid_pitch
        for poly in active_cell.polygons: poly.points = [(snap(p[0]), snap(p[1])) for p in poly.points]
        for ellipse in active_cell.ellipses:
            x, y, w, h = ellipse.rect
            # Snap position, ensure width/height are at least one grid pitch
            snapped_x, snapped_y = snap(x), snap(y)
            snapped_w = max(self.project.grid_pitch, round(w / self.project.grid_pitch) * self.project.grid_pitch)
            snapped_h = max(self.project.grid_pitch, round(h / self.project.grid_pitch) * self.project.grid_pitch)
            ellipse.rect = (snapped_x, snapped_y, snapped_w, snapped_h)
        for ref in active_cell.references: ref.origin = (snap(ref.origin[0]), snap(ref.origin[1]))
        self._save_state(); self._redraw_scene()

    def add_cell_dialog(self):
        if not self.project: return
        name, ok = QInputDialog.getText(self, "Add Cell", "Cell name:")
        if ok and name and name not in self.project.cells:
            self.project.cells[name] = Cell(); self._refresh_cell_list(); self._save_state()
    def rename_cell_dialog(self):
        if not self.project or not (sel := self.list_cells.selectedItems()): return
        old_name = sel[0].text()
        new_name, ok = QInputDialog.getText(self, "Rename Cell", "New name:", text=old_name)
        if ok and new_name and new_name != old_name and new_name not in self.project.cells:
            self.project.cells[new_name] = self.project.cells.pop(old_name)
            for cell in self.project.cells.values():
                for ref in cell.references:
                    if ref.cell == old_name: ref.cell = new_name
            if self.project.top == old_name: self.project.top = new_name
            if self.active_cell_name == old_name: self.active_cell_name = new_name
            self._refresh_cell_list(); self._redraw_scene(); self._save_state()
    def delete_cell_dialog(self):
        if not self.project or not (sel := self.list_cells.selectedItems()): return
        name = sel[0].text()
        if len(self.project.cells) <= 1: QMessageBox.warning(self, "Delete Error", "Cannot delete the last cell."); return
        if name == self.project.top: QMessageBox.warning(self, "Delete Error", "Cannot delete the top-level cell."); return
        
        # Check for references
        references_found_in = []
        for cell_name, cell in self.project.cells.items():
             if cell_name == name: continue # Don't check cell against itself
             if any(ref.cell == name for ref in cell.references):
                 references_found_in.append(cell_name)
        
        if references_found_in:
            QMessageBox.warning(self, "Delete Error", f"Cannot delete '{name}'. It is referenced by: {', '.join(references_found_in)}."); return
        
        # Proceed with deletion
        del self.project.cells[name]
        if self.active_cell_name == name: self.active_cell_name = self.project.top
        self._refresh_cell_list(); self._redraw_scene(); self._save_state()


    def add_layer_dialog(self):
        if not self.project: return
        name, ok = QInputDialog.getText(self, "Add Layer", "Layer name:")
        if ok and name and name not in self.project.layer_by_name:
            if (color := QColorDialog.getColor()).isValid():
                new_layer = Layer(name, (color.red(), color.green(), color.blue())) # Default fill='solid', thickness=10.0
                self.project.layers.append(new_layer); self.project.refresh_layer_map()
                self._refresh_layer_list(); self._save_state()
    def rename_layer_dialog(self):
        if not self.project or not (sel := self.list_layers.currentItem()): return
        old_name = sel.text()
        new_name, ok = QInputDialog.getText(self, "Rename Layer", "New name:", text=old_name)
        if ok and new_name and new_name != old_name and new_name not in self.project.layer_by_name:
            # Update all shapes
            for cell in self.project.cells.values():
                for p in cell.polygons:
                    if p.layer == old_name: p.layer = new_name
                for e in cell.ellipses:
                    if e.layer == old_name: e.layer = new_name
            # Update layer object itself
            layer_obj = self.project.layer_by_name.get(old_name)
            if layer_obj:
                layer_obj.name = new_name
                self.project.refresh_layer_map() # Crucial after name change
                if self.active_layer_name == old_name: self.active_layer_name = new_name
                self._refresh_layer_list(); self._redraw_scene(); self._save_state()
            else:
                QMessageBox.warning(self, "Rename Error", "Could not find layer object to rename.")


    def delete_layer_dialog(self):
        if not self.project or not (sel := self.list_layers.currentItem()): return
        name = sel.text()
        
        # Check for usage
        usage_found_in = []
        for cell_name, cell in self.project.cells.items():
            if any(p.layer == name for p in cell.polygons) or any(e.layer == name for e in cell.ellipses):
                usage_found_in.append(cell_name)
                break # No need to check further in this cell
        
        if usage_found_in:
             QMessageBox.warning(self, "Delete Error", f"Layer '{name}' is in use in cell '{usage_found_in[0]}'."); return

        # Proceed with deletion
        self.project.layers = [l for l in self.project.layers if l.name != name]; self.project.refresh_layer_map()
        if self.active_layer_name == name: self.active_layer_name = self.project.layers[0].name if self.project.layers else None
        self._refresh_layer_list(); self._save_state()


    def move_layer_forward(self): self._move_layer(1)
    def move_layer_backward(self): self._move_layer(-1)
    def move_layer_to_top(self):
        if self.project and (sel := self.list_layers.currentItem()) and (layer := self.project.layer_by_name.get(sel.text())):
            try:
                 idx = self.project.layers.index(layer)
                 if idx < len(self.project.layers) - 1:
                    self.project.layers.pop(idx); self.project.layers.append(layer)
                    self._save_state(); self._refresh_layer_list(); self._redraw_scene()
            except ValueError: return # Should not happen if layer is in layer_by_name

    def move_layer_to_bottom(self):
        if self.project and (sel := self.list_layers.currentItem()) and (layer := self.project.layer_by_name.get(sel.text())):
             try:
                idx = self.project.layers.index(layer)
                if idx > 0:
                    self.project.layers.pop(idx); self.project.layers.insert(0, layer)
                    self._save_state(); self._refresh_layer_list(); self._redraw_scene()
             except ValueError: return # Should not happen

    def _move_layer(self, direction):
        if not self.project or not (sel := self.list_layers.currentItem()): return
        if (layer := self.project.layer_by_name.get(sel.text())):
            try:
                idx = self.project.layers.index(layer); new_idx = idx + direction
                if 0 <= new_idx < len(self.project.layers):
                    self.project.layers.pop(idx); self.project.layers.insert(new_idx, layer)
                    self._save_state(); self._refresh_layer_list(); self._redraw_scene()
            except ValueError: return
    def change_layer_color_dialog(self, item):
        if self.project and (layer := self.project.layer_by_name.get(item.text())):
            if (color := QColorDialog.getColor(QColor(*layer.color))).isValid():
                layer.color = (color.red(), color.green(), color.blue())
                self._refresh_layer_list(); self._redraw_scene(); self._save_state()

    def _redraw_scene(self):
        if not hasattr(self, 'scene'): return # Guard against early calls
        self.scene.clear()
        if self.project and self.active_cell_name: self._draw_active_cell()
        if hasattr(self, 'view') and self.view and hasattr(self.view, 'MODES'):
            current_mode_name = next((name for name, val in self.view.MODES.items() if val == self.view.mode), "select")
            self.set_active_tool(current_mode_name)

    def _is_axis_aligned_rect(self, points):
        if len(points) != 4: return False
        try:
            xs, ys = set(p[0] for p in points), set(p[1] for p in points)
            return len(xs) == 2 and len(ys) == 2
        except Exception:
             return False # Handle potential errors if points is not as expected


    def _draw_active_cell(self):
        if not (cell := self.project.cells.get(self.active_cell_name)): return
        # Draw layers in order - bottom first
        for layer in self.project.layers:
            if not layer.visible: continue
            # Draw polygons on this layer
            for p_data in cell.polygons:
                if p_data.layer != layer.name: continue
                if self._is_axis_aligned_rect(p_data.points):
                    xs, ys = [p[0] for p in p_data.points], [p[1] for p in p_data.points]
                    self.scene.addItem(RectItem(QRectF(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)), layer, p_data))
                else:
                    self.scene.addItem(PolyItem(QPolygonF([QPointF(*pt) for pt in p_data.points]), layer, p_data))
            # Draw ellipses on this layer
            for e_data in cell.ellipses:
                if e_data.layer == layer.name: self.scene.addItem(CircleItem(QRectF(*e_data.rect), layer, e_data))
        # Draw references last (they contain items drawn according to layer order internally)
        for r_data in cell.references:
            if r_data.cell in self.project.cells: self.scene.addItem(RefItem(r_data, self.project.cells[r_data.cell], self.project))


    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, #ContentArea, QDockWidget { background-color: palette(window); color: palette(window-text); }
            QTabWidget::pane { border-top: 1px solid palette(mid); background: palette(base); }
            QTabBar { alignment: left; }
            QTabBar::tab { background: transparent; border: none; padding: 8px 15px; font-size: 9pt; color: palette(text); margin-right: 1px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:hover { background: palette(light); }
            QTabBar::tab:selected { background: palette(base); color: palette(highlight); }
            QToolButton { border: 1px solid transparent; padding: 5px; border-radius: 4px; color: palette(text); background-color: transparent; }
            QToolButton:hover { background-color: palette(light); border: 1px solid palette(mid); }
            QToolButton:pressed, QToolButton:checked { background-color: palette(midlight); border: 1px solid palette(mid); }
            QLabel, QCheckBox { color: palette(text); font-size: 8pt; }
            QFrame[frameShape="5"] { color: palette(midlight); }
        """)

    def get_sensible_default_value(self, divisor=100.0):
        """
        Calculates a reasonable default value (e.g., for widths, radii)
        based on the scale of the currently visible scene.
        """
        if not hasattr(self, 'view'):
            return 1.0

        try:
            visible_rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()

            if visible_rect.isEmpty() or visible_rect.width() <= 0:
                # Fallback if view/scene is empty or invalid
                 return self.project.grid_pitch if self.project and self.project.grid_pitch > 0 else 10.0


            # Use diagonal as a measure of scale, prevent division by zero
            diag = math.sqrt(visible_rect.width()**2 + visible_rect.height()**2)
            if diag <= 0: return 1.0

            default_value = diag / divisor # Base on diagonal, not just width

            # Clamp values to reasonable min/max
            if default_value < 0.001: return 0.001
            if default_value > 10000: return 10000

            return round(default_value, 3)
        except Exception:
            return 10.0 # Generic fallback

    def rename_selected_shape(self):
        if not self.project: return
        selected_items = self.scene.selectedItems()
        if len(selected_items) != 1: self.statusBar().showMessage("Please select a single shape to rename."); return
        item = selected_items[0]
        # Allow naming Refs as well
        target_obj = None
        if hasattr(item, 'data_obj'): # Poly, Ellipse, Rect
             if hasattr(item.data_obj, 'name'):
                 target_obj = item.data_obj
        # Cannot name Refs directly in the data model yet, skip for now
        # elif isinstance(item, RefItem):
        #    # Need to add 'name' field to Ref dataclass if desired
        #    pass

        if target_obj is None:
             self.statusBar().showMessage("Selected item cannot be named."); return

        current_name = target_obj.name or ""
        new_name, ok = QInputDialog.getText(self, "Rename Shape", "Enter name:", text=current_name)
        if ok:
            target_obj.name = new_name if new_name else None
            item.setToolTip(target_obj.name) # Update tooltip immediately
            self._save_state()

    def get_cell_bounds(self, cell_name):
        cell = self.project.cells.get(cell_name)
        if not cell: return QRectF()

        total_rect = QRectF()
        is_first = True

        def unite_rect(rect_to_add):
            nonlocal total_rect, is_first
            if rect_to_add.isNull() or not rect_to_add.isValid(): return # Skip invalid rects
            if is_first:
                total_rect = QRectF(rect_to_add) # Start with the first valid rect
                is_first = False
            else:
                total_rect = total_rect.united(rect_to_add)

        for p_data in cell.polygons:
            if not p_data.points: continue
            try:
                min_x = min(p[0] for p in p_data.points)
                max_x = max(p[0] for p in p_data.points)
                min_y = min(p[1] for p in p_data.points)
                max_y = max(p[1] for p in p_data.points)
                unite_rect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))
            except ValueError: continue # Handle empty points list case

        for e_data in cell.ellipses:
            unite_rect(QRectF(*e_data.rect))

        for ref in cell.references:
            # Prevent infinite recursion if a cell references itself
            if ref.cell == cell_name: continue
            child_bounds = self.get_cell_bounds(ref.cell)
            if not child_bounds.isNull():
                transform = QTransform().translate(ref.origin[0], ref.origin[1]).rotate(ref.rotation).scale(ref.magnification, ref.magnification)
                unite_rect(transform.mapRect(child_bounds))

        return total_rect

    def _run_boolean_op(self, op):
        if not self.project: return
        if not gdstk: QMessageBox.warning(self, "Feature Disabled", "Install 'gdstk'."); return
        selected = [it for it in self.scene.selectedItems() if isinstance(it, (PolyItem, RectItem, CircleItem))]
        if len(selected) < 2: self.statusBar().showMessage("Select at least two shapes for a boolean operation."); return
        first_layer = selected[0].layer.name
        if not all(it.layer.name == first_layer for it in selected):
            QMessageBox.warning(self, "Boolean Error", "All selected shapes must be on the same layer."); return
        gds_polys = []
        for item in selected:
            if isinstance(item, RectItem):
                r = item.rect()
                pts = [(r.left(), r.top()),(r.right(), r.top()),(r.right(), r.bottom()),(r.left(), r.bottom())]
                gds_polys.append(gdstk.Polygon(pts))
            elif isinstance(item, PolyItem):
                gds_polys.append(gdstk.Polygon([(p.x(), p.y()) for p in item.polygon()]))
            elif isinstance(item, CircleItem):
                r = item.rect()
                center = (r.center().x(), r.center().y())
                radius = (r.width() / 2, r.height() / 2)
                # gdstk.ellipse returns a Cell, need polygons from it
                temp_cell = gdstk.ellipse(center, (radius, radius)) # Use tuple for radius
                gds_polys.extend(temp_cell.polygons)

        if not gds_polys: # Handle case where only circles might have been selected but conversion failed
             QMessageBox.warning(self, "Boolean Error", "Could not convert selected shapes to polygons."); return

        try:
            # Ensure at least two operands for boolean
            if len(gds_polys) < 2:
                 QMessageBox.warning(self, "Boolean Error", "Need at least two valid polygons for operation."); return

            result_polys_or_cells = gdstk.boolean(gds_polys[0], gds_polys[1:], op)

            # gdstk.boolean returns a list of Polygons or sometimes Cells/RobustPaths, normalize to list of points
            final_result_points = []
            if result_polys_or_cells:
                 if isinstance(result_polys_or_cells, list) and len(result_polys_or_cells) > 0:
                    if isinstance(result_polys_or_cells[0], gdstk.Polygon):
                          final_result_points = [p.points.tolist() for p in result_polys_or_cells]
                 elif isinstance(result_polys_or_cells, gdstk.Polygon): # Handle single polygon return
                     final_result_points = [result_polys_or_cells.points.tolist()]
                 # Add handling for other potential return types if needed


        except Exception as e:
            QMessageBox.critical(self, "Boolean Operation Failed", str(e)); return

        active_cell = self.project.cells[self.active_cell_name]
        data_to_delete = [item.data_obj for item in selected]
        uuids_to_delete = {d.uuid for d in data_to_delete if hasattr(d, 'uuid')} # Check uuid exists
        active_cell.polygons = [p for p in active_cell.polygons if p.uuid not in uuids_to_delete]
        active_cell.ellipses = [e for e in active_cell.ellipses if e.uuid not in uuids_to_delete]

        for res_points in final_result_points:
             if res_points: # Ensure list is not empty
                 active_cell.polygons.append(Poly(layer=first_layer, points=res_points))

        self._save_state()
        self._redraw_scene()


    def copy_selected_shapes(self):
        if not self.project or not self.scene.selectedItems():
            return
        selected_data = []
        for item in self.scene.selectedItems():
            if hasattr(item, 'data_obj'):
                data_dict = asdict(item.data_obj)
                if isinstance(item.data_obj, Poly): data_dict['type'] = 'poly'
                elif isinstance(item.data_obj, Ellipse): data_dict['type'] = 'ellipse'
                # Do not copy Refs for now via clipboard, only shapes
                if 'type' in data_dict:
                     selected_data.append(data_dict)
        if selected_data:
            clipboard_data = {"layout_editor_clipboard": selected_data}
            QApplication.clipboard().setText(json.dumps(clipboard_data, cls=ProjectEncoder))
            self.statusBar().showMessage(f"Copied {len(selected_data)} shape(s).")


    def paste_shapes(self, scene_pos: Optional[QPointF] = None):
        if not self.project: return
        try:
            clipboard_text = QApplication.clipboard().text()
            data = json.loads(clipboard_text)
            if "layout_editor_clipboard" not in data: return
            pasted_items = data["layout_editor_clipboard"]
            active_cell = self.project.cells[self.active_cell_name]
            offset = self.project.grid_pitch if scene_pos is None else 0

            # Calculate center of pasted group
            min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
            has_valid_shape = False
            for item_data in pasted_items:
                if item_data.get('type') == 'poly' and item_data.get('points'):
                    has_valid_shape = True
                    for x, y in item_data['points']:
                        min_x, max_x = min(min_x, x), max(max_x, x)
                        min_y, max_y = min(min_y, y), max(max_y, y)
                elif item_data.get('type') == 'ellipse' and item_data.get('rect'):
                    has_valid_shape = True
                    x, y, w, h = item_data['rect']
                    min_x, max_x = min(min_x, x), max(max_x, x + w)
                    min_y, max_y = min(min_y, y), max(max_y, y + h)

            if not has_valid_shape: return # Nothing to paste

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            target_pos = self.view.snap_point(scene_pos) if scene_pos else QPointF(center_x + offset, center_y + offset)
            delta = target_pos - QPointF(center_x, center_y)

            newly_pasted_data = []
            for item_data in pasted_items:
                item_data.pop('uuid', None) # Remove old UUID
                item_type = item_data.pop('type', None)
                if item_type == 'poly' and item_data.get('points'):
                    item_data['points'] = [(p[0] + delta.x(), p[1] + delta.y()) for p in item_data['points']]
                    new_shape = Poly(**item_data)
                    active_cell.polygons.append(new_shape)
                    newly_pasted_data.append(new_shape)
                elif item_type == 'ellipse' and item_data.get('rect'):
                    x, y, w, h = item_data['rect']
                    item_data['rect'] = (x + delta.x(), y + delta.y(), w, h)
                    new_shape = Ellipse(**item_data)
                    active_cell.ellipses.append(new_shape)
                    newly_pasted_data.append(new_shape)

            self.statusBar().showMessage(f"Pasted {len(pasted_items)} shape(s).")
            self._save_state()
            self._redraw_scene()

            # Select newly pasted items
            self.scene.clearSelection()
            for item in self.scene.items():
                if hasattr(item, 'data_obj') and item.data_obj in newly_pasted_data:
                    item.setSelected(True)

        except (json.JSONDecodeError, TypeError, KeyError):
            self.statusBar().showMessage("Clipboard does not contain valid shape data.")

    def select_all_items(self):
        if hasattr(self, 'scene'):
            for item in self.scene.items():
                 # Only select items that are selectable
                 if item.flags() & QGraphicsItem.ItemIsSelectable:
                    item.setSelected(True)


    def zoom_in(self):
        if hasattr(self, 'view'): self.view.scale(1.2, 1.2); self.view.zoomChanged.emit()
    def zoom_out(self):
        if hasattr(self, 'view'): self.view.scale(1 / 1.2, 1 / 1.2); self.view.zoomChanged.emit()
    def show_all_layers(self):
        if not self.project: return
        changed = False
        for layer in self.project.layers:
             if not layer.visible:
                 layer.visible = True
                 changed = True
        if changed: self._refresh_layer_list(); self._redraw_scene(); self._save_state()

    def hide_all_layers(self):
        if not self.project: return
        changed = False
        for layer in self.project.layers:
             if layer.visible:
                 layer.visible = False
                 changed = True
        if changed: self._refresh_layer_list(); self._redraw_scene(); self._save_state()

    def toggle_rulers(self, checked):
        if hasattr(self, 'canvas_container'):
            self.canvas_container.h_ruler.setVisible(checked)
            self.canvas_container.v_ruler.setVisible(checked)

    def show_properties_dialog_for_item(self, item):
        """Opens the correct properties dialog for the given graphics item."""
        if not item: return

        target_obj = None
        kind = None
        params = {}

        if isinstance(item, RectItem) and hasattr(item, 'data_obj'):
            target_obj = item.data_obj
            r = item.rect()
            kind, params = "rect", {"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height()}
        elif isinstance(item, PolyItem) and hasattr(item, 'data_obj'):
             target_obj = item.data_obj
             pts = [(p.x(), p.y()) for p in item.polygon()]
             kind, params = "poly", {"points": pts}
        elif isinstance(item, CircleItem) and hasattr(item, 'data_obj'):
             target_obj = item.data_obj
             r = item.rect()
             params = {"center_x": r.center().x(), "center_y": r.center().y(), "w": r.width(), "h": r.height()}
             kind = "circle"
        # Add RefItem editing later if needed (Origin, Rotation, Mag)
        # elif isinstance(item, RefItem):
        #    target_obj = item.ref ...

        if kind and target_obj:
            dlg = ShapeEditDialog(kind, params, self.view)
            if dlg.exec_() == QDialog.Accepted and (vals := dlg.get_values()):
                original_data = copy.deepcopy(target_obj) # For undo
                try:
                    if kind == "rect":
                        item.setRect(QRectF(vals["x"], vals["y"], vals["w"], vals["h"]))
                        # Update data_obj points from new rect
                        r = item.rect()
                        target_obj.points = [(r.left(), r.top()), (r.right(), r.top()), (r.right(), r.bottom()), (r.left(), r.bottom())]
                    elif kind == "poly":
                        new_points = [QPointF(x, y) for x, y in vals["points"]]
                        if len(new_points) >= 3: # Basic validation
                            item.setPolygon(QPolygonF(new_points))
                            target_obj.points = vals["points"]
                        else:
                             QMessageBox.warning(self, "Invalid Polygon", "A polygon requires at least 3 vertices.")
                             return # Don't save invalid state
                    elif kind == "circle":
                        cx, cy, w, h = vals["center_x"], vals["center_y"], vals["w"], vals["h"]
                        item.setRect(QRectF(cx - w / 2, cy - h / 2, w, h))
                        target_obj.rect = (cx - w / 2, cy - h / 2, w, h)

                    # Update successful, save state
                    self._save_state()
                except Exception as e:
                     QMessageBox.critical(self, "Edit Error", f"Failed to apply changes: {e}")
                     # Attempt to restore original data if update failed
                     if isinstance(target_obj, Poly): target_obj.points = original_data.points
                     elif isinstance(target_obj, Ellipse): target_obj.rect = original_data.rect
                     self._redraw_scene() # Redraw to show restored state


    def can_paste(self):
        """Checks if the clipboard contains pastable shape data."""
        try:
            data = json.loads(QApplication.clipboard().text())
            return "layout_editor_clipboard" in data and isinstance(data["layout_editor_clipboard"], list)
        except (json.JSONDecodeError, TypeError, KeyError):
            return False

    def get_selection_bounds_and_center(self):
        """Calculates the bounding box and center of the current selection."""
        selected = self.scene.selectedItems()
        if not selected: return QRectF(), QPointF()
        bounds = QRectF()
        first = True
        for item in selected:
            item_bounds = item.sceneBoundingRect()
            if first:
                 bounds = item_bounds
                 first = False
            else:
                 bounds = bounds.united(item_bounds)
        return bounds, bounds.center()


    def _transform_shapes(self, transform_func):
        """Generic handler for transformations (rotate, scale, flip)."""
        if not self.project or not self.scene.selectedItems(): return

        active_cell = self.project.cells[self.active_cell_name]
        items_to_transform = self.scene.selectedItems()

        # Convert any selected ellipses to polygons for robust transformation
        new_polys_from_ellipses = []
        ellipses_to_remove_uuids = set()
        if gdstk: # Only convert if gdstk is available
            for item in items_to_transform:
                if isinstance(item, CircleItem):
                    ellipse_obj = item.data_obj
                    r = item.rect()
                    try:
                        # Use gdstk for a good circular approximation
                        gds_ellipse_cell = gdstk.ellipse((r.center().x(), r.center().y()), (r.width() / 2, r.height() / 2))
                        for poly in gds_ellipse_cell.polygons:
                            new_poly = Poly(layer=ellipse_obj.layer, name=ellipse_obj.name, points=poly.points.tolist())
                            new_polys_from_ellipses.append(new_poly)
                        ellipses_to_remove_uuids.add(ellipse_obj.uuid)
                    except Exception as e:
                         print(f"Warning: Could not convert ellipse to polygon: {e}")

        if new_polys_from_ellipses:
             # Remove old ellipses first
             active_cell.ellipses = [e for e in active_cell.ellipses if e.uuid not in ellipses_to_remove_uuids]
             active_cell.polygons.extend(new_polys_from_ellipses)
             # Redraw scene to get new PolyItems, then re-select them for transformation
             self._redraw_scene()
             all_items = self.scene.items()
             self.scene.clearSelection() # Clear old selection
             for scene_item in all_items:
                # Check data_obj and also check if it's one of the *newly created* polys
                if hasattr(scene_item, 'data_obj') and any(scene_item.data_obj is p for p in new_polys_from_ellipses):
                     scene_item.setSelected(True)


        # Apply the actual transformation logic
        transform_func() # This function modifies the data_obj/ref attributes

        self._save_state()
        self._redraw_scene() # Redraw scene from the modified data

    def rotate_selection(self):
        angle, ok = QInputDialog.getDouble(self, "Rotate Selection", "Angle (degrees):", 0, -360, 360, 1)
        if not ok: return

        def do_rotate():
            _, center = self.get_selection_bounds_and_center()
            transform = QTransform().translate(center.x(), center.y()).rotate(angle).translate(-center.x(), -center.y())

            for item in self.scene.selectedItems():
                # Handle Polygons (including converted Rects and Ellipses)
                if hasattr(item, 'data_obj') and isinstance(item.data_obj, Poly):
                    # Get the shape as a QPolygonF regardless of item type (PolyItem or RectItem)
                    if isinstance(item, PolyItem):
                        poly = item.polygon()
                    elif isinstance(item, RectItem):
                         # Convert rect to polygon for rotation
                        r = item.rect()
                        poly = QPolygonF([r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()])
                    else: continue

                    new_poly = transform.map(poly)
                    item.data_obj.points = [(p.x(), p.y()) for p in new_poly]

                # Handle Refs separately
                elif isinstance(item, RefItem):
                     # Rotate ref origin around selection center
                     ref_origin = QPointF(*item.ref.origin)
                     new_origin = transform.map(ref_origin)
                     item.ref.origin = (new_origin.x(), new_origin.y())
                     # Add rotation angle
                     item.ref.rotation = (item.ref.rotation + angle) % 360.0


        self._transform_shapes(do_rotate) # Use the wrapper

    def scale_selection(self):
        factor, ok = QInputDialog.getDouble(self, "Scale Selection", "Scale Factor:", 1.0, 0.01, 100.0, 2)
        if not ok or factor == 1.0: return

        def do_scale():
            _, center = self.get_selection_bounds_and_center()
            transform = QTransform().translate(center.x(), center.y()).scale(factor, factor).translate(-center.x(), -center.y())

            for item in self.scene.selectedItems():
                # Handle Polygons
                if hasattr(item, 'data_obj') and isinstance(item.data_obj, Poly):
                    if isinstance(item, PolyItem):
                        poly = item.polygon()
                    elif isinstance(item, RectItem):
                        r = item.rect()
                        poly = QPolygonF([r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()])
                    else: continue

                    new_poly = transform.map(poly)
                    item.data_obj.points = [(p.x(), p.y()) for p in new_poly]

                # Handle Refs separately
                elif isinstance(item, RefItem):
                     ref_origin = QPointF(*item.ref.origin)
                     new_origin = transform.map(ref_origin)
                     item.ref.origin = (new_origin.x(), new_origin.y())
                     item.ref.magnification *= factor


        self._transform_shapes(do_scale) # Use the wrapper

    def flip_selection_horizontal(self):
        def do_flip():
            _, center = self.get_selection_bounds_and_center()
            transform = QTransform().translate(center.x(), 0).scale(-1, 1).translate(-center.x(), 0)

            for item in self.scene.selectedItems():
                 # Handle Polygons
                if hasattr(item, 'data_obj') and isinstance(item.data_obj, Poly):
                    if isinstance(item, PolyItem):
                        poly = item.polygon()
                    elif isinstance(item, RectItem):
                        r = item.rect()
                        poly = QPolygonF([r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()])
                    else: continue

                    new_poly = transform.map(poly)
                    # Reverse point order for correct winding after flip if needed (gdstk handles this internally)
                    item.data_obj.points = [(p.x(), p.y()) for p in new_poly] #[::-1]

                 # Handle Refs separately
                elif isinstance(item, RefItem):
                     ref_origin = QPointF(*item.ref.origin)
                     new_origin = transform.map(ref_origin)
                     item.ref.origin = (new_origin.x(), new_origin.y())
                     item.ref.rotation = -item.ref.rotation % 360.0 # Flip rotation angle
                     # Need to handle magnification flip if x_reflection is supported by Ref


        self._transform_shapes(do_flip)

    def flip_selection_vertical(self):
        def do_flip():
            _, center = self.get_selection_bounds_and_center()
            transform = QTransform().translate(0, center.y()).scale(1, -1).translate(0, -center.y())

            for item in self.scene.selectedItems():
                 # Handle Polygons
                if hasattr(item, 'data_obj') and isinstance(item.data_obj, Poly):
                    if isinstance(item, PolyItem):
                        poly = item.polygon()
                    elif isinstance(item, RectItem):
                        r = item.rect()
                        poly = QPolygonF([r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()])
                    else: continue

                    new_poly = transform.map(poly)
                    item.data_obj.points = [(p.x(), p.y()) for p in new_poly] #[::-1]

                 # Handle Refs separately
                elif isinstance(item, RefItem):
                     ref_origin = QPointF(*item.ref.origin)
                     new_origin = transform.map(ref_origin)
                     item.ref.origin = (new_origin.x(), new_origin.y())
                     item.ref.rotation = (180.0 - item.ref.rotation) % 360.0 # Flip rotation angle


        self._transform_shapes(do_flip)


    def flip_selection(self):
        menu = QMenu(self)
        horiz = menu.addAction("Flip Horizontal")
        horiz.triggered.connect(self.flip_selection_horizontal)
        vert = menu.addAction("Flip Vertical")
        vert.triggered.connect(self.flip_selection_vertical)
        menu.exec_(QCursor.pos()) # Use global cursor pos

    def arrange_selection(self):
        menu = QMenu(self)
        front = menu.addAction("Bring to Front")
        front.triggered.connect(self.bring_selection_to_front)
        back = menu.addAction("Send to Back")
        back.triggered.connect(self.send_selection_to_back)
        menu.exec_(QCursor.pos()) # Use global cursor pos

    def move_selection_forward(self):
        if not (cell := self.project.cells.get(self.active_cell_name)) or not self.scene.selectedItems(): return
        selected_data = {item.data_obj for item in self.scene.selectedItems() if hasattr(item, 'data_obj')}
        if not selected_data: return

        # Move polygons forward
        moved = False
        poly_list = cell.polygons
        for i in range(len(poly_list) - 2, -1, -1):
            if poly_list[i] in selected_data and poly_list[i+1] not in selected_data:
                poly_list[i], poly_list[i+1] = poly_list[i+1], poly_list[i]
                moved = True

        # Move ellipses forward
        ellipse_list = cell.ellipses
        for i in range(len(ellipse_list) - 2, -1, -1):
            if ellipse_list[i] in selected_data and ellipse_list[i+1] not in selected_data:
                ellipse_list[i], ellipse_list[i+1] = ellipse_list[i+1], ellipse_list[i]
                moved = True
        
        # Note: Doesn't move Refs
        
        if moved: self._save_state(); self._redraw_scene()

    def move_selection_backward(self):
        if not (cell := self.project.cells.get(self.active_cell_name)) or not self.scene.selectedItems(): return
        selected_data = {item.data_obj for item in self.scene.selectedItems() if hasattr(item, 'data_obj')}
        if not selected_data: return

        moved = False
        # Move polygons backward
        poly_list = cell.polygons
        for i in range(1, len(poly_list)):
            if poly_list[i] in selected_data and poly_list[i-1] not in selected_data:
                poly_list[i], poly_list[i-1] = poly_list[i-1], poly_list[i]
                moved = True

        # Move ellipses backward
        ellipse_list = cell.ellipses
        for i in range(1, len(ellipse_list)):
            if ellipse_list[i] in selected_data and ellipse_list[i-1] not in selected_data:
                ellipse_list[i], ellipse_list[i-1] = ellipse_list[i-1], ellipse_list[i]
                moved = True

        if moved: self._save_state(); self._redraw_scene()

    def bring_selection_to_front(self):
        if not (cell := self.project.cells.get(self.active_cell_name)) or not self.scene.selectedItems(): return
        selected_data = {item.data_obj for item in self.scene.selectedItems() if hasattr(item, 'data_obj')}
        if not selected_data: return

        polys_to_move = [p for p in cell.polygons if p in selected_data]
        ellipses_to_move = [e for e in cell.ellipses if e in selected_data]

        cell.polygons = [p for p in cell.polygons if p not in selected_data] + polys_to_move
        cell.ellipses = [e for e in cell.ellipses if e not in selected_data] + ellipses_to_move

        self._save_state(); self._redraw_scene()

    def send_selection_to_back(self):
        if not (cell := self.project.cells.get(self.active_cell_name)) or not self.scene.selectedItems(): return
        selected_data = {item.data_obj for item in self.scene.selectedItems() if hasattr(item, 'data_obj')}
        if not selected_data: return

        polys_to_move = [p for p in cell.polygons if p in selected_data]
        ellipses_to_move = [e for e in cell.ellipses if e in selected_data]

        cell.polygons = polys_to_move + [p for p in cell.polygons if p not in selected_data]
        cell.ellipses = ellipses_to_move + [e for e in cell.ellipses if e not in selected_data]

        self._save_state(); self._redraw_scene()

    def move_selection_to_layer(self, layer_name):
        selected = self.scene.selectedItems()
        if not selected: return
        target_layer = self.project.layer_by_name.get(layer_name)
        if not target_layer: return # Target layer doesn't exist

        changed = False
        for item in selected:
            if hasattr(item, 'data_obj') and item.data_obj.layer != layer_name:
                item.data_obj.layer = layer_name
                changed = True

        if changed: self._save_state(); self._redraw_scene()

    def measure_selection(self):
        selected = self.scene.selectedItems()
        if not selected or not gdstk: return

        total_area, total_perimeter = 0.0, 0.0

        for item in selected:
            try:
                gds_shape = None
                if isinstance(item, PolyItem):
                    gds_shape = gdstk.Polygon([(p.x(), p.y()) for p in item.polygon()])
                elif isinstance(item, RectItem):
                    r = item.rect()
                    pts = [(r.left(), r.top()), (r.right(), r.top()), (r.right(), r.bottom()), (r.left(), r.bottom())]
                    gds_shape = gdstk.Polygon(pts)
                elif isinstance(item, CircleItem):
                    r = item.rect()
                    # gdstk.ellipse returns a Cell, get polygon from it
                    temp_cell = gdstk.ellipse((r.center().x(), r.center().y()), (r.width() / 2, r.height() / 2))
                    if temp_cell.polygons:
                         gds_shape = temp_cell.polygons[0] # Assume first polygon is representative for area/perimeter

                if gds_shape:
                     total_area += gds_shape.area()
                     total_perimeter += gds_shape.perimeter()

            except Exception as e:
                print(f"Warning: Measurement failed for an item. {e}")

        QMessageBox.information(self, "Measurement",
            f"Selected items: {len(selected)}\n"
            f"Total Area: {total_area:.3f} sq. {self.project.units}\n"
            f"Total Perimeter: {total_perimeter:.3f} {self.project.units}")

# -------------------------- main --------------------------
def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("MyCompany")
    app.setApplicationName("2D Mask Layout Editor")
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    icons_dir = os.path.join(script_dir, 'icons')
    if not os.path.exists(icons_dir):
        os.makedirs(icons_dir)
        print("Created 'icons' directory. Please populate it with SVG icons for the best experience.")
    welcome_dialog = WelcomeDialog()
    if not welcome_dialog.exec_() == QDialog.Accepted:
        sys.exit(0)
    win = MainWindow()
    win.show()
    if welcome_dialog.choice == 'open':
        win.open_file(path=welcome_dialog.open_path)
    elif welcome_dialog.choice == 'new':
        win.new_doc()
    sys.exit(app.exec_())

class ProjectEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, UUID): return str(o)
        if is_dataclass(o): return asdict(o)
        # Handle numpy types
        if _has_level_set_deps:
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8,
                               np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                return float(o)
            elif isinstance(o, (np.ndarray,)):
                return o.tolist()
            elif isinstance(o, np.bool_):
                 return bool(o)
        return super().default(o)


if __name__ == "__main__":
    main()
