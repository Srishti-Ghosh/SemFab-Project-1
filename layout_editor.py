import sys
import math
import json
import copy
import uuid
import os
from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Dict, List, Tuple, Optional
from uuid import UUID

# This is a stub for the gdstk library if it's not installed.
# For full functionality, install it: pip install gdstk
try:
    import gdstk
except ImportError:
    print("Warning: gdstk library not found. GDS/OAS, fillet, and boolean features will be disabled.")
    gdstk = None

# Added for the 3D view logic
try:
    from matplotlib.path import Path
except ImportError:
    print("Warning: matplotlib library not found. 3D view will be disabled.")
    Path = None

from PyQt5.QtCore import Qt, QRectF, QPointF, QSize, QMimeData, pyqtSignal, QLineF
from PyQt5.QtGui import (QBrush, QPen, QColor, QPolygonF, QPainter, QPixmap, QIcon,
                         QDrag, QPalette, QFont, QTransform)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QColorDialog,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPolygonItem,
    QGraphicsItem, QToolBar, QPushButton, QLabel, QLineEdit,
    QSpinBox, QCheckBox, QDialog, QVBoxLayout, QDialogButtonBox, QToolTip,
    QDockWidget, QListWidget, QListWidgetItem, QWidget, QHBoxLayout, QMessageBox, QDoubleSpinBox, QGraphicsEllipseItem,
    QInputDialog, QTabWidget, QFrame, QToolButton, QGridLayout, QSlider, QGraphicsLineItem
)
from PyQt5.QtSvg import QSvgGenerator

# -------------------------- Data model --------------------------

@dataclass
class Layer:
    name: str
    color: Tuple[int, int, int]
    visible: bool = True

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
    """A professional welcome screen for the application."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to the Layout Editor")
        self.setMinimumSize(450, 300)
        self.choice = None
        self.open_path = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("2D Mask Layout Editor")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; padding-bottom: 10px;")
        layout.addWidget(title)

        btn_new = QPushButton(self._create_themed_icon("file-plus", 24), " Create New Project")
        btn_new.setIconSize(QSize(24, 24))
        btn_new.setStyleSheet("padding: 10px; text-align: left; font-size: 14px;")
        btn_new.clicked.connect(self.new_project)
        layout.addWidget(btn_new)

        btn_open = QPushButton(self._create_themed_icon("folder", 24), " Open Existing File...")
        btn_open.setIconSize(QSize(24, 24))
        btn_open.setStyleSheet("padding: 10px; text-align: left; font-size: 14px;")
        btn_open.clicked.connect(self.open_file)
        layout.addWidget(btn_open)

        layout.addStretch()

    def _create_themed_icon(self, icon_name, size=32):
        # Fallback to current working directory if __file__ is not available (e.g., in an interactive environment)
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fillet Polygon")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Fillet Radius:"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 1000.0); self.radius_spin.setValue(10.0)
        layout.addWidget(self.radius_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_radius(self): return self.radius_spin.value()

class Interactive3DView(QGraphicsView):
    def __init__(self, scene, parent_dialog):
        super().__init__(scene)
        self.parent_dialog = parent_dialog
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self._last_pan_point = QPointF()

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
            self.parent_dialog.draw_3d_view()
        self._last_pan_point = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        self.parent_dialog.zoom_scale *= zoom_factor
        self.parent_dialog.draw_3d_view()

class ThreeDViewDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive 3D Preview")
        self.setGeometry(100, 100, 800, 800)
        self.project = project
        self.angle_x, self.angle_y, self.zoom_scale = 35.0, -45.0, 1.0
        self.layer_thicknesses = {layer.name: 10 for layer in self.project.layers}
        self.sliders = {}

        main_layout = QVBoxLayout(self)
        top_panel = QWidget()
        top_panel_layout = QHBoxLayout(top_panel)
        top_panel_layout.setContentsMargins(0, 0, 0, 0)
        sliders_frame = QFrame(); sliders_frame.setFrameShape(QFrame.StyledPanel)
        sliders_layout = QGridLayout(sliders_frame)
        sliders_layout.addWidget(QLabel("<b>Layer Thickness</b>"), 0, 0, 1, 3)
        row = 1
        for layer in self.project.layers:
            label, slider = QLabel(layer.name), QSlider(Qt.Horizontal)
            slider.setRange(1, 100); slider.setValue(self.layer_thicknesses.get(layer.name, 10))
            slider.valueChanged.connect(self.on_slider_change)
            value_label = QLabel(f"{slider.value()}")
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
        btn_iso, btn_top = QPushButton("3D / Iso"), QPushButton("Top")
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
        self.view = Interactive3DView(self.scene, self)
        main_layout.addWidget(self.view)
        self.draw_3d_view()

    def set_view_angles(self, x, y): self.angle_x, self.angle_y = x, y; self.draw_3d_view()
    def set_iso_view(self): self.set_view_angles(35.0, -45.0)
    def set_top_view(self): self.set_view_angles(0, 0)
    def set_front_view(self): self.set_view_angles(-90, 0)
    def set_side_view(self): self.set_view_angles(0, -90)

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "3d_view.png", "PNG (*.png);;JPG (*.jpg)")
        if path and not self.view.grab().save(path):
            QMessageBox.warning(self, "Save Error", f"Could not save image to:\n{path}")

    def on_slider_change(self):
        for name, slider in self.sliders.items():
            self.layer_thicknesses[name] = slider.value()
            slider.value_label.setText(f"{slider.value()}")
        self.draw_3d_view()

    def project_point(self, x, y, z):
        rad_y, rad_x = math.radians(self.angle_y), math.radians(self.angle_x)
        cos_y, sin_y, cos_x, sin_x = math.cos(rad_y), math.sin(rad_y), math.cos(rad_x), math.sin(rad_x)
        x1 = x * cos_y + z * sin_y; z1 = -x * sin_y + z * cos_y
        y1 = y * cos_x - z1 * sin_x
        return QPointF(x1 * self.zoom_scale, -y1 * self.zoom_scale)

    def draw_3d_view(self):
        self.scene.clear()

        def get_all_shapes_as_polygons(cell, origin=(0, 0), rotation=0, mag=1.0):
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

        all_shapes_by_layer = {layer.name: [] for layer in self.project.layers}
        all_shapes = get_all_shapes_as_polygons(self.project.cells[self.project.top])
        if not all_shapes: return

        for layer_name, points in all_shapes:
            if layer_name in all_shapes_by_layer:
                all_shapes_by_layer[layer_name].append(points)

        all_points_flat = [pt for _, poly_pts in all_shapes for pt in poly_pts]
        if not all_points_flat: return

        min_x, max_x = min(p[0] for p in all_points_flat), max(p[0] for p in all_points_flat)
        min_y, max_y = min(p[1] for p in all_points_flat), max(p[1] for p in all_points_flat)
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

        grid_step = 25.0
        height_map, all_faces = {}, []

        for layer in self.project.layers:
            if not layer.visible or not Path: continue

            thickness = self.layer_thicknesses.get(layer.name, 10)
            color = QColor(*layer.color)

            for points_2d in all_shapes_by_layer.get(layer.name, []):
                path = Path(points_2d)
                
                p_min_x_g = int(min(p[0] for p in points_2d) // grid_step)
                p_max_x_g = int(max(p[0] for p in points_2d) // grid_step)
                p_min_y_g = int(min(p[1] for p in points_2d) // grid_step)
                p_max_y_g = int(max(p[1] for p in points_2d) // grid_step)

                footprint_z_values = [0]
                for gx in range(p_min_x_g, p_max_x_g + 1):
                    for gy in range(p_min_y_g, p_max_y_g + 1):
                        if path.contains_point(((gx + 0.5) * grid_step, (gy + 0.5) * grid_step)):
                            footprint_z_values.append(height_map.get((gx, gy), 0))
                
                base_z = max(footprint_z_values)

                centered_pts = [(p[0] - center_x, p[1] - center_y) for p in points_2d]
                floor_pts = [(*p, base_z) for p in centered_pts]
                ceiling_pts = [(*p, base_z + thickness) for p in centered_pts]
                all_faces.append((ceiling_pts, color.lighter(110)))
                all_faces.append((floor_pts, color.darker(120)))

                for i in range(len(floor_pts)):
                    p1, p2 = floor_pts[i], floor_pts[(i + 1) % len(floor_pts)]
                    p3, p4 = ceiling_pts[(i + 1) % len(floor_pts)], ceiling_pts[i]
                    all_faces.append(([p1, p2, p3, p4], color))

                new_top_z = base_z + thickness
                for gx in range(p_min_x_g, p_max_x_g + 1):
                    for gy in range(p_min_y_g, p_max_y_g + 1):
                        if path.contains_point(((gx + 0.5) * grid_step, (gy + 0.5) * grid_step)):
                            height_map[(gx, gy)] = new_top_z

        total_height = max(height_map.values()) if height_map else 0

        # --- NEW CODE START: Add a translucent bounding box ---
        if total_height > 0:
            w = max_x - min_x
            h = max_y - min_y
            d = total_height

            # Define the 8 vertices of the box, centered like the model
            v = [
                (-w/2, -h/2, 0), (w/2, -h/2, 0), (w/2, h/2, 0), (-w/2, h/2, 0), # Bottom
                (-w/2, -h/2, d), (w/2, -h/2, d), (w/2, h/2, d), (-w/2, h/2, d)  # Top
            ]

            # Define the 6 faces using vertex indices
            box_faces_indices = [
                (0, 1, 2, 3),  # Bottom
                (4, 5, 6, 7),  # Top
                (0, 3, 7, 4),  # Left
                (1, 2, 6, 5),  # Right
                (0, 1, 5, 4),  # Back
                (2, 3, 7, 6)   # Front
            ]
            
            box_color = QColor(150, 150, 150, 40) # A light, translucent gray
            for face_indices in box_faces_indices:
                face_points = [v[i] for i in face_indices]
                all_faces.append((face_points, box_color))
        # --- NEW CODE END ---

        all_faces.sort(key=lambda face: sum(self.project_point(*p).y() for p in face[0])/len(face[0]))

        for points_3d, col in all_faces:
            points_2d = [self.project_point(*p) for p in points_3d]
            pen = QPen(col.darker(110), 0)
            self.scene.addPolygon(QPolygonF(points_2d), pen, QBrush(col))
            
        axis_length = max(max_x - min_x, max_y - min_y, total_height) * 0.75
        origin_proj = self.project_point(0, 0, 0)
        axes = [((axis_length,0,0),"red","X"), ((0,axis_length,0),"green","Y"), ((0,0,axis_length),"blue","Z")]
        for axis, color_str, name in axes:
            end = self.project_point(*axis)
            self.scene.addLine(QLineF(origin_proj, end), QPen(QColor(color_str), 2))
            label = self.scene.addText(name); label.setDefaultTextColor(QColor(color_str)); label.setPos(end)
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.view.setSceneRect(self.scene.itemsBoundingRect().adjusted(-50, -50, 50, 50))


# -------------------------- Graphics items --------------------------

class VertexHandle(QGraphicsEllipseItem):
    def __init__(self, parent_poly, index):
        self.poly_item = parent_poly
        self.index = index
        point = self.poly_item.polygon()[self.index]
        super().__init__(point.x() - 4, point.y() - 4, 8, 8, parent=parent_poly)
        self.setBrush(QBrush(QColor("cyan")))
        self.setPen(QPen(QColor("blue"), 0.5))
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            new_pos = self.poly_item.mapToParent(value)
            self.poly_item.update_vertex(self.index, new_pos)
        return super().itemChange(change, value)

class SceneItemMixin:
    """A mixin to handle shared mouse press/release logic for draggable items."""
    def __init__(self, *args, **kwargs):
        self._drag_start_pos = QPointF()

    def customMousePressEvent(self, event):
        """Custom logic to be called by implementing classes."""
        self._drag_start_pos = self.pos()

    def customMouseReleaseEvent(self, event):
        """Custom logic to be called by implementing classes."""
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsItem.ItemIsMovable:
            if self.pos() != self._drag_start_pos:
                self.scene().views()[0].parent_win.update_data_from_item_move(self)

class RectItem(QGraphicsRectItem, SceneItemMixin):
    def __init__(self, rect, layer, data_obj, selectable=True):
        QGraphicsRectItem.__init__(self, rect)
        SceneItemMixin.__init__(self)
        self.layer = layer; self.base_color = QColor(*self.layer.color); self.data_obj = data_obj
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable if selectable else 0)
        self.refresh_appearance(selected=False)
        if self.data_obj.name: self.setToolTip(self.data_obj.name)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.customMousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.customMouseReleaseEvent(event)

    def refresh_appearance(self, selected=False):
        color = self.base_color.lighter(130) if selected else self.base_color
        color.setAlpha(180)
        self.setBrush(QBrush(color)); self.setPen(QPen(Qt.black, 0.5))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self.refresh_appearance(selected=bool(value))
        elif change == QGraphicsItem.ItemPositionChange and self.scene():
            view = self.scene().views()[0]
            if view.parent_win.chk_snap.isChecked():
                original_top_left = self.rect().topLeft()
                proposed_scene_top_left = original_top_left + value
                snapped_scene_top_left = view.snap_point(proposed_scene_top_left)
                return snapped_scene_top_left - original_top_left
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event); r = self.rect()
        dlg = ShapeEditDialog("rect", {"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height()}, self.scene().views()[0])
        if dlg.exec_() == QDialog.Accepted and (vals := dlg.get_values()):
            self.setRect(QRectF(vals["x"], vals["y"], vals["w"], vals["h"]))
            self.scene().views()[0].parent_win.update_data_from_item_edit(self)

class PolyItem(QGraphicsPolygonItem, SceneItemMixin):
    def __init__(self, poly, layer, data_obj, selectable=True):
        QGraphicsPolygonItem.__init__(self, poly)
        SceneItemMixin.__init__(self)
        self.layer = layer; self.base_color = QColor(*self.layer.color); self.data_obj = data_obj
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable if selectable else 0)
        self.refresh_appearance(selected=False)
        self.handles = []
        if self.data_obj.name: self.setToolTip(self.data_obj.name)

    def mousePressEvent(self, event):
        win = self.scene().views()[0].parent_win
        if win.view.mode == win.view.MODES["vertex_edit"]:
            win.start_vertex_edit(self)
            event.accept()
        else:
            super().mousePressEvent(event)
            self.customMousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.customMouseReleaseEvent(event)

    def refresh_appearance(self, selected=False):
        color = self.base_color.lighter(130) if selected else self.base_color
        color.setAlpha(180)
        self.setBrush(QBrush(color)); self.setPen(QPen(Qt.black, 0.5))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self.refresh_appearance(selected=bool(value))
        elif change == QGraphicsItem.ItemPositionChange and self.scene():
            view = self.scene().views()[0]
            if view.parent_win.chk_snap.isChecked():
                original_top_left = self.polygon().boundingRect().topLeft()
                proposed_scene_top_left = original_top_left + value
                snapped_scene_top_left = view.snap_point(proposed_scene_top_left)
                return snapped_scene_top_left - original_top_left
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event); pts = [(p.x(), p.y()) for p in self.polygon()]
        dlg = ShapeEditDialog("poly", {"points": pts}, self.scene().views()[0])
        if dlg.exec_() == QDialog.Accepted and (vals := dlg.get_values()):
            self.setPolygon(QPolygonF([QPointF(x, y) for x, y in vals["points"]]))
            self.scene().views()[0].parent_win.update_data_from_item_edit(self)

    def update_vertex(self, index, pos):
        poly = self.polygon()
        poly[index] = self.mapFromScene(pos) if self.scene() else pos
        self.setPolygon(poly)
        self.scene().views()[0].parent_win.update_data_from_item_edit(self)

class CircleItem(QGraphicsEllipseItem, SceneItemMixin):
    def __init__(self, rect, layer, data_obj, selectable=True):
        QGraphicsEllipseItem.__init__(self, rect)
        SceneItemMixin.__init__(self)
        self.layer = layer; self.base_color = QColor(*self.layer.color); self.data_obj = data_obj
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable if selectable else 0)
        self.refresh_appearance(selected=False)
        if self.data_obj.name: self.setToolTip(self.data_obj.name)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.customMousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.customMouseReleaseEvent(event)

    def refresh_appearance(self, selected=False):
        color = self.base_color.lighter(130) if selected else self.base_color
        color.setAlpha(180); self.setBrush(QBrush(color)); self.setPen(QPen(Qt.black, 0.5))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self.refresh_appearance(selected=bool(value))
        elif change == QGraphicsItem.ItemPositionChange and self.scene():
            view = self.scene().views()[0]
            if view.parent_win.chk_snap.isChecked():
                original_top_left = self.rect().topLeft()
                proposed_scene_top_left = original_top_left + value
                snapped_scene_top_left = view.snap_point(proposed_scene_top_left)
                return snapped_scene_top_left - original_top_left
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event); r = self.rect()
        params = {"center_x": r.center().x(), "center_y": r.center().y(), "w": r.width(), "h": r.height()}
        dlg = ShapeEditDialog("circle", params, self.scene().views()[0])
        if dlg.exec_() == QDialog.Accepted and (vals := dlg.get_values()):
            cx, cy, w, h = vals["center_x"], vals["center_y"], vals["w"], vals["h"]
            self.setRect(QRectF(cx - w / 2, cy - h / 2, w, h))
            self.scene().views()[0].parent_win.update_data_from_item_edit(self)

class RefItem(QGraphicsItem):
    def __init__(self, ref, cell, project, selectable=True):
        super().__init__()
        self.ref, self.cell, self.project = ref, cell, project
        self._drag_start_pos = QPointF()
        if selectable: self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.setPos(QPointF(*self.ref.origin))
        self.setRotation(self.ref.rotation); self.setScale(self.ref.magnification)
        self._draw_children()

    def _draw_children(self):
        for item in self.childItems(): item.setParentItem(None)
        for p_data in self.cell.polygons:
            if (layer := self.project.layer_by_name.get(p_data.layer)) and layer.visible:
                PolyItem(QPolygonF([QPointF(*pt) for pt in p_data.points]), layer, p_data, False).setParentItem(self)
        for e_data in self.cell.ellipses:
            if (layer := self.project.layer_by_name.get(e_data.layer)) and layer.visible:
                CircleItem(QRectF(*e_data.rect), layer, e_data, False).setParentItem(self)
        for r_data in self.cell.references:
            if r_data.cell in self.project.cells:
                RefItem(r_data, self.project.cells[r_data.cell], self.project, selectable=False).setParentItem(self)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self._drag_start_pos = self.pos()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsItem.ItemIsMovable:
            if self.pos() != self._drag_start_pos:
                self.scene().views()[0].parent_win.update_data_from_item_move(self)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            view = self.scene().views()[0]
            if view.parent_win.chk_snap.isChecked():
                return view.snap_point(value)
        return super().itemChange(change, value)

    def boundingRect(self): return self.childrenBoundingRect() or QRectF(-5, -5, 10, 10)

    def paint(self, painter, option, widget):
        if self.isSelected():
            painter.setPen(QPen(Qt.darkCyan, 0, Qt.DashLine))
            painter.setBrush(Qt.NoBrush); painter.drawRect(self.boundingRect())

# -------------------------- View/Scene with Rulers --------------------------

class Ruler(QWidget):
    def __init__(self, orientation, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas; self.orientation = orientation
        self.ruler_font = QFont("Arial", 8)
        self.setFixedHeight(30) if orientation == Qt.Horizontal else self.setFixedWidth(30)

    def paintEvent(self, event):
        if not self.canvas.parent_win.project: return
        painter = QPainter(self); painter.setFont(self.ruler_font)
        bg_color, tick_color = self.palette().color(QPalette.Window), self.palette().color(QPalette.WindowText)
        painter.fillRect(self.rect(), bg_color)
        visible_rect = self.canvas.mapToScene(self.canvas.viewport().rect()).boundingRect()
        zoom = self.canvas.transform().m11()
        base_pitch = self.canvas.parent_win.project.grid_pitch
        if base_pitch <= 0: return
        minor_pitch = float(base_pitch); required_screen_pitch = 8
        while (minor_pitch * zoom) < required_screen_pitch: minor_pitch *= 5
        while (minor_pitch * zoom) > required_screen_pitch * 5: minor_pitch /= 5
        painter.setPen(QPen(tick_color, 1))
        if self.orientation == Qt.Horizontal:
            start = math.floor(visible_rect.left() / minor_pitch) * minor_pitch
            end = math.ceil(visible_rect.right() / minor_pitch) * minor_pitch
            count = int(round(start / minor_pitch)); x = start
            while x <= end:
                view_x = self.canvas.mapFromScene(QPointF(x, 0)).x()
                tick_height = 10 if count % 5 == 0 else 5
                painter.drawLine(view_x, self.height(), view_x, self.height() - tick_height)
                if count % 5 == 0 and zoom > 0.05:
                    painter.drawText(view_x + 2, self.height() - 12, f"{x:.0f}")
                x += minor_pitch; count += 1
        else: # Vertical
            start = math.floor(visible_rect.top() / minor_pitch) * minor_pitch
            end = math.ceil(visible_rect.bottom() / minor_pitch) * minor_pitch
            count = int(round(start / minor_pitch)); y = start
            while y <= end:
                view_y = self.canvas.mapFromScene(QPointF(0, y)).y()
                tick_width = 10 if count % 5 == 0 else 5
                painter.drawLine(self.width(), view_y, self.width() - tick_width, view_y)
                if count % 5 == 0 and zoom > 0.05:
                    painter.save()
                    painter.translate(self.width() - 12, view_y - 2); painter.rotate(90)
                    painter.drawText(0, 0, f"{-y:.0f}") # Y is inverted on canvas
                    painter.restore()
                y += minor_pitch; count += 1

class CanvasContainer(QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.h_ruler = Ruler(Qt.Horizontal, self.canvas)
        self.v_ruler = Ruler(Qt.Vertical, self.canvas)
        layout = QGridLayout(); layout.setSpacing(0); layout.setContentsMargins(0,0,0,0)
        corner = QWidget(); corner.setFixedSize(30, 30); corner.setStyleSheet("background-color: palette(window);")
        layout.addWidget(self.h_ruler, 0, 1); layout.addWidget(self.v_ruler, 1, 0)
        layout.addWidget(self.canvas, 1, 1); layout.addWidget(corner, 0, 0)
        self.setLayout(layout)
        self.canvas.horizontalScrollBar().valueChanged.connect(self.h_ruler.update)
        self.canvas.verticalScrollBar().valueChanged.connect(self.v_ruler.update)
        self.canvas.zoomChanged.connect(self.h_ruler.update)
        self.canvas.zoomChanged.connect(self.v_ruler.update)

class Canvas(QGraphicsView):
    MODES = {"select": 1, "rect": 2, "poly": 3, "circle": 4, "move": 5, "measure": 6, "vertex_edit": 7}
    zoomChanged = pyqtSignal()

    def __init__(self, scene, get_active_layer, parent=None):
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing); self.setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse); self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.mode = self.MODES["select"]; self.start_pos = None; self.temp_item = None
        self.temp_poly_points: List[QPointF] = []
        self.get_active_layer = get_active_layer; self.parent_win = parent
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._panning = False

    def drawBackground(self, painter: QPainter, rect: QRectF):
        super().drawBackground(painter, rect)
        if not self.parent_win.project: return
        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        zoom = self.transform().m11()
        bg_color, minor_color, major_color = self.palette().color(self.backgroundRole()), self.palette().color(self.foregroundRole()), self.palette().color(self.foregroundRole())
        minor_color.setAlphaF(0.2); major_color.setAlphaF(0.4)
        painter.fillRect(rect, bg_color)
        base_pitch = self.parent_win.project.grid_pitch
        if base_pitch <= 0: return
        minor_pitch = float(base_pitch); required_screen_pitch = 8
        while (minor_pitch * zoom) < required_screen_pitch: minor_pitch *= 5
        while (minor_pitch * zoom) > required_screen_pitch * 5: minor_pitch /= 5
        major_pitch = minor_pitch * 5
        minor_pen, major_pen = QPen(minor_color, 0), QPen(major_color, 0)
        left, right, top, bottom = visible_rect.left(), visible_rect.right(), visible_rect.top(), visible_rect.bottom()
        x_start = math.floor(left / minor_pitch) * minor_pitch; x_end = math.ceil(right / minor_pitch) * minor_pitch
        count = int(round(x_start / minor_pitch)); x = x_start
        while x <= x_end:
            painter.setPen(major_pen if count % 5 == 0 else minor_pen)
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            x += minor_pitch; count += 1
        y_start = math.floor(top / minor_pitch) * minor_pitch; y_end = math.ceil(bottom / minor_pitch) * minor_pitch
        count = int(round(y_start / minor_pitch)); y = y_start
        while y <= y_end:
            painter.setPen(major_pen if count % 5 == 0 else minor_pen)
            painter.drawLine(QPointF(left, y), QPointF(right, y))
            y += minor_pitch; count += 1

    def wheelEvent(self, event):
        zoom_in_factor = 1.15; zoom_out_factor = 1 / zoom_in_factor
        old_pos = self.mapToScene(event.pos())
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom_factor, zoom_factor)
        new_pos = self.mapToScene(event.pos()); delta = new_pos - old_pos
        self.translate(delta.x(), delta.y()); self.zoomChanged.emit()

    def set_mode(self, mode_name: str):
        self.parent_win.stop_vertex_edit()
        self.mode = self.MODES.get(mode_name, self.MODES["select"])

        # --- FIX 1: RESTORED MOVE LOGIC ---
        is_move_mode = (self.mode == self.MODES["move"])
        is_select_mode = (self.mode == self.MODES["select"])
        # --- End of Fix ---

        if is_select_mode:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
        for item in self.scene().items():
            if item.flags() & QGraphicsItem.ItemIsSelectable:
                # --- FIX 1 (cont.): USE CORRECT FLAGS ---
                item.setFlag(QGraphicsItem.ItemIsMovable, is_select_mode or is_move_mode)
                # --- End of Fix ---
        self.temp_cancel()

    def temp_cancel(self):
        if self.temp_item: self.scene().removeItem(self.temp_item)
        self.temp_item = None; self.temp_poly_points.clear()
        self.viewport().setCursor(Qt.ArrowCursor)

    def snap_point(self, p: QPointF) -> QPointF:
        if not self.parent_win.chk_snap.isChecked(): return p
        pitch = self.parent_win.project.grid_pitch
        if pitch <= 0: return p
        return QPointF(round(p.x() / pitch) * pitch, round(p.y() / pitch) * pitch)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True; self._pan_start = event.pos(); self.setCursor(Qt.ClosedHandCursor)
            event.accept(); return

        scene_pos = self.snap_point(self.mapToScene(event.pos()))
        if self.mode in [self.MODES["rect"], self.MODES["circle"]] and event.button() == Qt.LeftButton:
            self.start_pos = scene_pos
            self.temp_item = QGraphicsRectItem(QRectF(self.start_pos, self.start_pos)) if self.mode == self.MODES["rect"] else QGraphicsEllipseItem(QRectF(self.start_pos, self.start_pos))
            self.temp_item.setPen(QPen(Qt.gray, 0, Qt.DashLine)); self.scene().addItem(self.temp_item)
            event.accept(); return
        if self.mode == self.MODES["measure"] and event.button() == Qt.LeftButton:
            self.start_pos = scene_pos
            self.temp_item = QGraphicsLineItem(QLineF(self.start_pos, self.start_pos))
            self.temp_item.setPen(QPen(QColor(255,100,0), 2, Qt.DashLine)); self.scene().addItem(self.temp_item)
            event.accept(); return
        if self.mode == self.MODES["poly"] and event.button() == Qt.LeftButton:
            self.temp_poly_points.append(scene_pos)
            if self.temp_item: self.scene().removeItem(self.temp_item)
            if len(self.temp_poly_points) >= 1:
                poly = QPolygonF(self.temp_poly_points)
                if len(self.temp_poly_points) > 1: self.temp_item = self.scene().addPolygon(poly, QPen(Qt.gray, 0, Qt.DashLine))
                else: self.temp_item = self.scene().addRect(poly.boundingRect(), QPen(Qt.gray, 0, Qt.DashLine))
            event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if self.mode != self.MODES["measure"]:
            self.parent_win.statusBar().showMessage(f"X: {scene_pos.x():.2f}, Y: {-scene_pos.y():.2f}")
        if self._panning:
            delta = self._pan_start - event.pos(); self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
            event.accept(); return
        if self.temp_item and self.start_pos:
            snapped_pos = self.snap_point(scene_pos)
            if self.mode in [self.MODES["rect"], self.MODES["circle"]]:
                self.temp_item.setRect(QRectF(self.start_pos, snapped_pos).normalized())
            elif self.mode == self.MODES["measure"]:
                self.temp_item.setLine(QLineF(self.start_pos, snapped_pos))
                dx = snapped_pos.x() - self.start_pos.x()
                dy = snapped_pos.y() - self.start_pos.y()
                dist = math.sqrt(dx**2 + dy**2)
                self.parent_win.statusBar().showMessage(f"dX: {dx:.2f}, dY: {-dy:.2f}, Dist: {dist:.2f}")
            event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False; self.setCursor(Qt.ArrowCursor); event.accept(); return
        if self.mode in [self.MODES["rect"], self.MODES["circle"]] and self.start_pos:
            self.temp_cancel()
            rect = QRectF(self.start_pos, self.snap_point(self.mapToScene(event.pos()))).normalized()
            if rect.width() > 0 and rect.height() > 0 and (layer := self.get_active_layer()):
                if self.mode == self.MODES["rect"]: self.parent_win.add_rect_to_active_cell(rect, layer)
                else: self.parent_win.add_circle_to_active_cell(rect, layer)
            self.start_pos = None; event.accept(); return
        if self.mode == self.MODES["measure"] and self.start_pos:
            self.temp_cancel(); self.start_pos = None; event.accept(); return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.mode == self.MODES["poly"] and event.button() == Qt.LeftButton:
            self.finish_polygon(); event.accept(); return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.parent_win.stop_vertex_edit()
            self.parent_win.set_active_tool("select")
            self.temp_cancel()
        elif event.key() == Qt.Key_Delete: self.parent_win.delete_selected_items()
        else: super().keyPressEvent(event)

    def finish_polygon(self):
        can_create_poly = len(self.temp_poly_points) >= 3 and (layer := self.get_active_layer())
        if can_create_poly: final_poly = QPolygonF(self.temp_poly_points)
        self.temp_cancel()
        if can_create_poly: self.parent_win.add_poly_to_active_cell(final_poly, layer)

    def dragEnterEvent(self, event): event.acceptProposedAction() if event.mimeData().hasText() else event.ignore()
    def dragMoveEvent(self, event): event.acceptProposedAction() if event.mimeData().hasText() else event.ignore()

    def dropEvent(self, event):
        cell_name = event.mimeData().text()
        if not (cell_name and cell_name in self.parent_win.project.cells):
            event.ignore(); return
        if cell_name == self.parent_win.active_cell_name:
            QToolTip.showText(event.globalPos(), "Cannot drop a cell onto itself."); event.ignore(); return
        item_at_pos = self.itemAt(event.pos()); target_data_obj = None
        if hasattr(item_at_pos, 'data_obj') and (item_at_pos.flags() & QGraphicsItem.ItemIsSelectable):
            if isinstance(item_at_pos, RectItem) or \
               (isinstance(item_at_pos, PolyItem) and self.parent_win._is_axis_aligned_rect(item_at_pos.data_obj.points)):
                target_data_obj = item_at_pos.data_obj
        if target_data_obj:
            self.parent_win.instantiate_cell_on_shape(cell_name, target_data_obj)
        else:
            scene_pos = self.mapToScene(event.pos())
            self.parent_win.add_reference_to_active_cell(cell_name, self.snap_point(scene_pos))
        event.acceptProposedAction()


# -------------------------- Main window --------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = None
        self.current_file_path = None
        self._is_dirty = False
        self._vertex_edit_item = None
        self._undo_stack, self._redo_stack = [], []

        self._build_ui()
        self._apply_stylesheet()
        self.setCentralWidget(QLabel("Create a new project or open a file to begin.", alignment=Qt.AlignCenter))
        self._update_window_title()
        self.statusBar().showMessage("Welcome!")
        self.resize(1600, 1000)

    def initialize_project(self, project):
        self.project = project
        self.active_cell_name = self.project.top
        self.active_layer_name = self.project.layers[0].name if self.project.layers else None

        self.scene = QGraphicsScene(0, 0, self.project.canvas_width, self.project.canvas_height)
        self.view = Canvas(self.scene, self.active_layer, self)
        self.view.scale(1, -1)

        self.canvas_container = CanvasContainer(self.view, self)
        self.setCentralWidget(self.canvas_container)

        self.update_ui_from_project()
        self._undo_stack, self._redo_stack = [], []
        self._save_state()
        self._is_dirty = False
        self._update_window_title()
        self.fill_view()

    def _create_default_project(self, pitch, width, height):
        return Project(layers=[Layer("N-doping", (0, 0, 255)), Layer("P-doping", (255, 0, 0)), Layer("Oxide", (0, 200, 0)), Layer("Metal", (200, 200, 200)), Layer("Contact", (255, 200, 0))],
                       cells={"TOP": Cell()}, top="TOP",
                       grid_pitch=pitch, canvas_width=width, canvas_height=height)

    def _mark_dirty(self):
        if not self._is_dirty:
            self._is_dirty = True
            self._update_window_title()

    def _update_window_title(self):
        title = "2D Mask Layout Editor"
        filename = os.path.basename(self.current_file_path) if self.current_file_path else "Untitled"
        self.setWindowTitle(f"{filename}[*] - {title}")
        self.setWindowModified(self._is_dirty)

    def closeEvent(self, event):
        if self._is_dirty:
            reply = QMessageBox.question(self, 'Unsaved Changes',
                                       "You have unsaved changes. Do you want to save them before closing?",
                                       QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                       QMessageBox.Save)
            if reply == QMessageBox.Save:
                if self.save_json():
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def _prompt_save_if_dirty(self):
        if not self._is_dirty:
            return True
        reply = QMessageBox.question(self, 'Unsaved Changes',
                                   "You have unsaved changes. Do you want to save them first?",
                                   QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                   QMessageBox.Save)
        if reply == QMessageBox.Save:
            return self.save_json()
        elif reply == QMessageBox.Discard:
            return True
        else:
            return False

    def new_doc(self):
        if not self._prompt_save_if_dirty():
            return
        grid_dialog = GridDialog(self)
        if grid_dialog.exec_() == QDialog.Accepted:
            pitch, w, h = grid_dialog.get_values()
            self.current_file_path = None
            self.initialize_project(self._create_default_project(pitch, w, h))

    def open_file(self, path=None):
        if not self._prompt_save_if_dirty():
            return
        if not path:
            file_filter = "Layout Files (*.json *.gds *.oas);;JSON Project (*.json);;GDSII Files (*.gds);;OASIS Files (*.oas)"
            path, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter)
        if not path:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.json':
                self._load_json(path)
            elif ext in ['.gds', '.oas']:
                self._load_gds_oas(path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            self.statusBar().showMessage(f"Successfully opened {path}")
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to open file:\n{e}")
            self.statusBar().showMessage(f"Failed to open {path}")
        finally:
            QApplication.restoreOverrideCursor()

    def _load_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        project = Project(
            units=data.get('units', 'um'), top=data.get('top', 'TOP'),
            layers=[Layer(**l) for l in data.get('layers', [])],
            grid_pitch=data.get('grid_pitch', 50),
            canvas_width=data.get('canvas_width', 2000),
            canvas_height=data.get('canvas_height', 1500)
        )
        for name, c_data in data.get('cells', {}).items():
            for p in c_data.get('polygons', []): p.pop('uuid', None)
            for e in c_data.get('ellipses', []): e.pop('uuid', None)
            project.cells[name] = Cell(polygons=[Poly(**p) for p in c_data.get('polygons', [])],
                                       ellipses=[Ellipse(**e) for e in c_data.get('ellipses', [])],
                                       references=[Ref(**r) for r in c_data.get('references', [])])
        project.refresh_layer_map()
        self.current_file_path = path
        self.initialize_project(project)

    def _load_gds_oas(self, path):
        if not gdstk: raise ImportError("GDSTK library is required for this feature.")
        lib = gdstk.read_gds(path) if path.lower().endswith('.gds') else gdstk.read_oas(path)

        sidecar_path = path + '.json'
        layer_meta = {}
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, 'r') as f:
                    sidecar_data = json.load(f)
                    layer_meta = {int(k): v for k, v in sidecar_data.get('layer_metadata', {}).items()}
            except Exception as e:
                print(f"Warning: Could not read sidecar metadata file. {e}")

        project = self._create_default_project(50, 2000, 1500)
        project.layers.clear()
        project.cells.clear()

        gds_layers = sorted(list(lib.layers))
        for l_num in gds_layers:
            if l_num in layer_meta:
                meta = layer_meta[l_num]
                project.layers.append(Layer(name=meta['name'], color=tuple(meta['color']), visible=meta['visible']))
            else:
                project.layers.append(Layer(f"Layer_{l_num}", (l_num*20%255, l_num*50%255, l_num*80%255)))

        project.refresh_layer_map()
        layer_num_to_name = {l_num: f"Layer_{l_num}" for l_num in gds_layers}
        for l_num, meta in layer_meta.items():
             if 'name' in meta: layer_num_to_name[l_num] = meta['name']

        for cell in lib.cells:
            new_cell = Cell()
            for poly in cell.polygons:
                new_cell.polygons.append(Poly(layer_num_to_name.get(poly.layer, f"Layer_{poly.layer}"), poly.points.tolist()))
            for ref in cell.references:
                ref_cell_name = ref.cell.name if isinstance(ref.cell, gdstk.Cell) else ref.cell
                new_cell.references.append(Ref(ref_cell_name, ref.origin, ref.rotation, ref.magnification))
            project.cells[cell.name] = new_cell

        project.top = lib.top_level()[0].name if lib.top_level() else (list(project.cells.keys())[0] if project.cells else None)
        self.current_file_path = None
        self.initialize_project(project)

    def save_json(self):
        if self.current_file_path:
            return self._write_json(self.current_file_path)
        else:
            return self.save_json_as()

    def save_json_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON As", self.current_file_path or "", "JSON Files (*.json)")
        if path:
            return self._write_json(path)
        return False

    def _write_json(self, path):
        if not self.project: return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            class ProjectEncoder(json.JSONEncoder):
                def default(self, o):
                    if isinstance(o, UUID): return str(o)
                    if is_dataclass(o): return asdict(o)
                    return super().default(o)
            with open(path, "w") as f:
                json.dump(self.project, f, indent=2, cls=ProjectEncoder)

            self.current_file_path = path
            self._is_dirty = False
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
        if not gdstk:
            QMessageBox.warning(self, "Feature Disabled", "Install 'gdstk'.")
            return
        file_ext = "*.oas" if is_oas else "*.gds"
        path, _ = QFileDialog.getSaveFileName(self, f"Save {file_ext.upper()}", "", f"Files ({file_ext})")
        if not path:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            lib = gdstk.Library(unit=float(self.project.units.replace('um','e-6')))
            layer_name_to_num = {layer.name: i for i, layer in enumerate(self.project.layers, 1)}

            gds_cells = {name: lib.new_cell(name) for name in self.project.cells}
            for name, cell_data in self.project.cells.items():
                gds_cell = gds_cells[name]
                for p in cell_data.polygons: gds_cell.add(gdstk.Polygon(p.points, layer=layer_name_to_num.get(p.layer, 0)))
                for e in cell_data.ellipses:
                    r = e.rect; center = (r[0] + r[2]/2, r[1] + r[3]/2); radius = (r[2]/2)
                    gds_cell.add(gdstk.ellipse(center, radius, layer=layer_name_to_num.get(e.layer, 0)))
                for r in cell_data.references:
                    if r.cell in gds_cells:
                        gds_cell.add(gdstk.Reference(gds_cells[r.cell], r.origin, r.rotation, r.magnification))

            lib.write_oas(path) if is_oas else lib.write_gds(path)

            layer_metadata = {
                layer_name_to_num[l.name]: {'name': l.name, 'color': l.color, 'visible': l.visible}
                for l in self.project.layers if l.name in layer_name_to_num
            }
            sidecar_path = path + '.json'
            with open(sidecar_path, 'w') as f:
                json.dump({'layer_metadata': layer_metadata}, f, indent=2)

            self.statusBar().showMessage(f"Exported to {path} and created metadata file.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to write file: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    # --- UI Building & Core Logic ---

    def _create_themed_icon(self, icon_name):
        try:
            # Fallback to current working directory if __file__ is not available (e.g., in an interactive environment)
            script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
            icon_path = os.path.join(script_dir, 'icons', f'{icon_name}.svg')
            pixmap = QPixmap(icon_path)
            is_dark = self.palette().color(QPalette.Window).value() < 128
            if is_dark:
                painter = QPainter(pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                painter.fillRect(pixmap.rect(), Qt.white); painter.end()
            return QIcon(pixmap)
        except Exception:
            pixmap = QPixmap(32, 32); pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap); painter.setPen(QPen(Qt.gray, 2))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, icon_name[0].upper()); painter.end()
            return QIcon(pixmap)

    def _build_ui(self):
        self.tool_buttons = {}
        self._build_menus(); self._build_tool_tabs()
        self._build_cell_dock(); self._build_layer_dock()
        self.set_active_tool("select")

    def _build_menus(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction("New", self.new_doc, "Ctrl+N")
        file_menu.addAction("Open...", self.open_file, "Ctrl+O")
        file_menu.addAction("Save", self.save_json, "Ctrl+S")
        file_menu.addAction("Save As...", self.save_json_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        if gdstk:
            file_menu.addAction("Export GDS…", lambda: self._write_gdstk(False))
            file_menu.addAction("Export OAS…", lambda: self._write_gdstk(True))
        file_menu.addAction("Export SVG…", self.export_svg)
        file_menu.addSeparator()
        file_menu.addAction("Undo", self.undo, "Ctrl+Z")
        file_menu.addAction("Redo", self.redo, "Ctrl+Y")
        file_menu.addSeparator(); file_menu.addAction("Quit", self.close, "Ctrl+Q")

    def _build_tool_tabs(self):
        dock = QDockWidget(self); dock.setTitleBarWidget(QWidget()); dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetMovable); tabs = QTabWidget(); dock.setWidget(tabs)

        main_tab = QWidget(); main_layout = QHBoxLayout(main_tab)
        main_layout.setAlignment(Qt.AlignLeft); main_layout.setContentsMargins(0, 5, 0, 5)
        main_layout.addWidget(self._create_ribbon_group("File", [
            self._create_action_button("Save", "save", self.save_json),
            self._create_action_button("Open", "folder", self.open_file),
            self._create_action_button("Refresh", "refresh-cw", self._redraw_scene)
        ]))
        main_layout.addWidget(self._create_ribbon_group("Selection", [
            self._create_tool_button("select", "Select", "mouse-pointer"),
            self._create_tool_button("move", "Move", "move")
        ]))
        main_layout.addWidget(self._create_ribbon_group("History", [
            self._create_action_button("Delete", "trash-2", self.delete_selected_items),
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
        main_layout.addStretch(); tabs.addTab(main_tab, "Main")

        draw_tab = QWidget(); draw_layout = QHBoxLayout(draw_tab)
        draw_layout.setAlignment(Qt.AlignLeft); draw_layout.setContentsMargins(0, 5, 0, 5)
        draw_layout.addWidget(self._create_ribbon_group("Create", [
            self._create_tool_button("rect", "Rect", "square"),
            self._create_tool_button("circle", "Circle", "circle"),
            self._create_tool_button("poly", "Poly", "pen-tool")
        ]))
        draw_layout.addWidget(self._create_ribbon_group("Modify", [
            self._create_action_button("Rename", "tag", self.rename_selected_shape),
            self._create_tool_button("vertex_edit", "Vertex Edit", "git-pull-request"),
            self._create_action_button("Re-snap", "grid", self.resnap_all_items_to_grid),
            self._create_action_button("Finish Poly", "check-square", lambda: self.view.finish_polygon()),
            self._create_action_button("Fillet", "git-commit", self.fillet_selected_poly)
        ]))
        draw_layout.addWidget(self._create_ribbon_group("Boolean", [
            self._create_action_button("Union", "plus-square", lambda: self._run_boolean_op('or')),
            self._create_action_button("Subtract", "minus-square", lambda: self._run_boolean_op('not')),
            self._create_action_button("Intersect", "crop", lambda: self._run_boolean_op('and'))
        ]))
        draw_layout.addStretch(); tabs.addTab(draw_tab, "Draw")

        view_tab = QWidget(); view_layout = QHBoxLayout(view_tab)
        view_layout.setAlignment(Qt.AlignLeft); view_layout.setContentsMargins(0, 5, 0, 5)
        view_layout.addWidget(self._create_ribbon_group("Display", [
            self._create_action_button("Fill", "maximize", self.fill_view),
            self._create_action_button("3D View", "box", self.show_3d_view),
            self._create_tool_button("measure", "Measure", "compass")
        ]))
        view_layout.addStretch(); tabs.addTab(view_tab, "View")
        self.addDockWidget(Qt.TopDockWidgetArea, dock)

    def _create_tool_button(self, tool_name, text, icon_name):
        button = QToolButton(); button.setText(text); button.setIcon(self._create_themed_icon(icon_name))
        button.setIconSize(QSize(24, 24)); button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setCheckable(True); button.clicked.connect(lambda: self.set_active_tool(tool_name))
        self.tool_buttons[tool_name] = button
        return button

    def _create_action_button(self, text, icon_name, slot):
        button = QToolButton(); button.setText(text); button.setIcon(self._create_themed_icon(icon_name))
        button.setIconSize(QSize(24, 24)); button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.clicked.connect(slot)
        return button

    def _create_ribbon_group(self, title, widgets):
        group_container = QWidget(); group_layout = QHBoxLayout(group_container)
        group_layout.setContentsMargins(0, 0, 0, 0); group_layout.setSpacing(0)
        buttons_widget = QWidget(); buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(10, 5, 10, 5); buttons_layout.setSpacing(5)
        panel_widget = QWidget(); panel_layout = QVBoxLayout(panel_widget)
        panel_layout.setContentsMargins(0,0,0,0); panel_layout.setSpacing(2); panel_layout.setAlignment(Qt.AlignBottom)
        for widget in widgets: buttons_layout.addWidget(widget)
        title_label = QLabel(title); title_label.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(buttons_widget); panel_layout.addWidget(title_label)
        group_layout.addWidget(panel_widget); separator = QFrame()
        separator.setFrameShape(QFrame.VLine); separator.setFrameShadow(QFrame.Sunken)
        group_layout.addWidget(separator)
        return group_container

    def set_active_tool(self, tool_name):
        if hasattr(self, 'view') and self.view: self.view.set_mode(tool_name)
        for name, btn in self.tool_buttons.items(): btn.setChecked(name == tool_name)

    def _build_cell_dock(self):
        dock = QDockWidget("Cells", self); self.list_cells = QListWidget()
        self.list_cells.setDragEnabled(True); self.list_cells.itemDoubleClicked.connect(self._on_cell_double_clicked)
        widget, layout = QWidget(), QVBoxLayout()
        layout.addWidget(QLabel("Double-click to edit, Drag to instantiate")); layout.addWidget(self.list_cells)
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
        if self.project.top not in self.project.cells: self.project.top = list(self.project.cells.keys())[0] if self.project.cells else None
        if not self.active_cell_name or self.active_cell_name not in self.project.cells: self.active_cell_name = self.project.top
        self.spin_grid.setValue(self.project.grid_pitch)
        self._refresh_cell_list(); self._refresh_layer_list(); self._redraw_scene()

    def _refresh_cell_list(self):
        self.list_cells.blockSignals(True); current_text = self.active_cell_name; self.list_cells.clear()
        for name in sorted(self.project.cells.keys()):
            item = QListWidgetItem(name); self.list_cells.addItem(item)
            if name == current_text: item.setSelected(True)
        self.list_cells.blockSignals(False)

    def _refresh_layer_list(self):
        self.list_layers.blockSignals(True); current_text = self.active_layer_name; self.list_layers.clear()
        for layer in self.project.layers:
            item = QListWidgetItem(layer.name); item.setIcon(self._create_color_icon(QColor(*layer.color)))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable); item.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
            self.list_layers.addItem(item)
            if layer.name == current_text: self.list_layers.setCurrentItem(item)
        self.list_layers.blockSignals(False)

    def _create_color_icon(self, color, size=16):
        pixmap = QPixmap(size, size); pixmap.fill(Qt.transparent)
        p = QPainter(pixmap); p.setBrush(QBrush(color)); p.setPen(Qt.black)
        p.drawRect(0, 0, size - 1, size - 1); p.end()
        return QIcon(pixmap)

    def _on_cell_double_clicked(self, item):
        if item.text() != self.active_cell_name: self.active_cell_name = item.text(); self._redraw_scene()
    def _on_layer_selected(self):
        if item := self.list_layers.currentItem(): self.active_layer_name = item.text()
    def _update_layer_visibility(self, item):
        if (layer := self.project.layer_by_name.get(item.text())):
            layer.visible = item.checkState() == Qt.Checked; self._redraw_scene(); self._save_state()

    def active_layer(self): return self.project.layer_by_name.get(self.active_layer_name) if self.project else None

    def _save_state(self):
        if not self.project: return
        self._redo_stack.clear()
        self._undo_stack.append(copy.deepcopy(self.project))
        if len(self._undo_stack) > 50: self._undo_stack.pop(0)
        if len(self._undo_stack) > 1:
             self._mark_dirty()

    def undo(self):
        if len(self._undo_stack) > 1:
            self._redo_stack.append(self._undo_stack.pop())
            self.project = copy.deepcopy(self._undo_stack[-1])
            self.update_ui_from_project()
            self._mark_dirty()
    def redo(self):
        if self._redo_stack:
            self.project = self._redo_stack.pop()
            self._undo_stack.append(copy.deepcopy(self.project))
            self.update_ui_from_project()
            self._mark_dirty()

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
    def add_reference_to_active_cell(self, cell_name, pos):
        if cell_name == self.active_cell_name: return
        self.project.cells[self.active_cell_name].references.append(Ref(cell_name, (pos.x(), pos.y())))
        self._redraw_scene(); self._save_state()

    def instantiate_cell_on_shape(self, cell_name_to_drop, target_shape_data):
        if isinstance(target_shape_data, Poly):
            xs = [p[0] for p in target_shape_data.points]; ys = [p[1] for p in target_shape_data.points]
            placeholder_rect = QRectF(QPointF(min(xs), min(ys)), QPointF(max(xs), max(ys)))
        elif isinstance(target_shape_data, Ellipse): placeholder_rect = QRectF(*target_shape_data.rect)
        else: return
        dropped_cell_bounds = self.get_cell_bounds(cell_name_to_drop); magnification = 1.0
        if not dropped_cell_bounds.isNull() and dropped_cell_bounds.width() > 0 and dropped_cell_bounds.height() > 0:
            scale_x = placeholder_rect.width() / dropped_cell_bounds.width()
            scale_y = placeholder_rect.height() / dropped_cell_bounds.height()
            magnification = min(scale_x, scale_y)
        origin = placeholder_rect.center() - (dropped_cell_bounds.center() * magnification)
        active_cell = self.project.cells[self.active_cell_name]
        if isinstance(target_shape_data, Poly): active_cell.polygons = [p for p in active_cell.polygons if p.uuid != target_shape_data.uuid]
        elif isinstance(target_shape_data, Ellipse): active_cell.ellipses = [e for e in active_cell.ellipses if e.uuid != target_shape_data.uuid]
        active_cell.references.append(Ref(cell=cell_name_to_drop, origin=(origin.x(), origin.y()), magnification=magnification))
        self._save_state(); self._redraw_scene()

    # --- FIX 2: RESTORED WORKING MOVE UPDATE LOGIC ---
    def update_data_from_item_move(self, item):
        drag_start_pos = item._drag_start_pos
        # The item's final position is already snapped by its itemChange method
        final_pos = item.pos()
        dx = final_pos.x() - drag_start_pos.x()
        dy = final_pos.y() - drag_start_pos.y()

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return

        # Update the underlying data model based on the delta
        if isinstance(item, RefItem):
            item.ref.origin = (item.ref.origin[0] + dx, item.ref.origin[1] + dy)
        elif isinstance(item, CircleItem):
            r = item.data_obj.rect
            item.data_obj.rect = (r[0] + dx, r[1] + dy, r[2], r[3])
        elif isinstance(item, (RectItem, PolyItem)):
            item.data_obj.points = [(p[0] + dx, p[1] + dy) for p in item.data_obj.points]

        # The item's visual position is already correct. We just need to save the state.
        # No need to redraw or reset the item's position.
        self._save_state()
        self._redraw_scene() # Redraw to ensure all states are consistent
    # --- End of Fix ---

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
        if not self.project: return
        cell = self.project.cells.get(self.active_cell_name)
        if not cell or not (selected_items := self.scene.selectedItems()): return
        data_objects_to_delete = [item.data_obj for item in selected_items if hasattr(item, 'data_obj')]
        data_objects_to_delete.extend([item.ref for item in selected_items if isinstance(item, RefItem)])
        if data_objects_to_delete:
            self.scene.clearSelection()
            cell.polygons = [p for p in cell.polygons if p not in data_objects_to_delete]
            cell.ellipses = [e for e in cell.ellipses if e not in data_objects_to_delete]
            cell.references = [r for r in cell.references if r not in data_objects_to_delete]
            self._redraw_scene(); self._save_state()

    def fillet_selected_poly(self):
        if not gdstk: QMessageBox.warning(self, "Feature Disabled", "Please install 'gdstk' to use this feature."); return
        selected = [it for it in self.scene.selectedItems() if isinstance(it, PolyItem)]
        if not selected: self.statusBar().showMessage("Select one or more polygons to fillet."); return
        dlg = FilletDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            radius = dlg.get_radius()
            for item in selected:
                try:
                    gds_poly = gdstk.Polygon([(p.x(), p.y()) for p in item.polygon()]); gds_poly.fillet(radius)
                    item.data_obj.points = gds_poly.points.tolist()
                except Exception as e: QMessageBox.warning(self, "Fillet Error", f"Could not fillet polygon: {e}")
            self._save_state(); self._redraw_scene()

    def show_measurements(self):
        # This is a placeholder for the old show_measurements, which has been removed.
        # The new logic is in Canvas.mouseMoveEvent for the measure tool.
        # This can be removed or reimplemented if static measurement is needed.
        pass

    def fill_view(self):
        if not self.project or not self.scene.items(): return
        items_rect = self.scene.itemsBoundingRect()
        if not items_rect.isNull():
            margin = max(items_rect.width(), items_rect.height()) * 0.05
            self.view.fitInView(items_rect.adjusted(-margin, -margin, margin, margin), Qt.KeepAspectRatio)
            self.view.zoomChanged.emit()

    def show_3d_view(self):
        if self.project:
            ThreeDViewDialog(self.project, self).exec_()

    def export_svg(self):
        if not self.project: return
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", "", "SVG Files (*.svg)")
        if not path: return
        gen = QSvgGenerator()
        gen.setFileName(path)
        rect = self.scene.itemsBoundingRect().adjusted(-50, -50, 50, 50)
        gen.setViewBox(rect)
        painter = QPainter(gen)
        self.scene.render(painter, target=QRectF(), source=rect)
        painter.end()

    def set_grid_pitch(self, value):
        if not self.project: return
        self.project.grid_pitch = value
        self.view.viewport().update()
        self._mark_dirty()

    def resnap_all_items_to_grid(self):
        if not self.project: return
        reply = QMessageBox.question(self, 'Re-snap Cell', "This will snap all shapes in the current cell to the grid. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No or self.project.grid_pitch <= 0: return
        active_cell = self.project.cells.get(self.active_cell_name)
        if not active_cell: return
        def snap(val): return round(val / self.project.grid_pitch) * self.project.grid_pitch
        for poly in active_cell.polygons: poly.points = [(snap(p[0]), snap(p[1])) for p in poly.points]
        for ellipse in active_cell.ellipses:
            x, y, w, h = ellipse.rect
            ellipse.rect = (snap(x), snap(y), max(self.project.grid_pitch, snap(w)), max(self.project.grid_pitch, snap(h)))
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
        for cell_name, cell in self.project.cells.items():
            if any(ref.cell == name for ref in cell.references):
                QMessageBox.warning(self, "Delete Error", f"Cannot delete '{name}' as it is referenced by '{cell_name}'."); return
        del self.project.cells[name]
        if self.active_cell_name == name: self.active_cell_name = self.project.top
        self._refresh_cell_list(); self._redraw_scene(); self._save_state()

    def add_layer_dialog(self):
        if not self.project: return
        name, ok = QInputDialog.getText(self, "Add Layer", "Layer name:")
        if ok and name and name not in self.project.layer_by_name:
            if (color := QColorDialog.getColor()).isValid():
                new_layer = Layer(name, (color.red(), color.green(), color.blue()))
                self.project.layers.append(new_layer); self.project.refresh_layer_map()
                self._refresh_layer_list(); self._save_state()
    def rename_layer_dialog(self):
        if not self.project or not (sel := self.list_layers.currentItem()): return
        old_name = sel.text()
        new_name, ok = QInputDialog.getText(self, "Rename Layer", "New name:", text=old_name)
        if ok and new_name and new_name != old_name and new_name not in self.project.layer_by_name:
            for cell in self.project.cells.values():
                for p in cell.polygons:
                    if p.layer == old_name: p.layer = new_name
                for e in cell.ellipses:
                    if e.layer == old_name: e.layer = new_name
            self.project.layer_by_name[old_name].name = new_name; self.project.refresh_layer_map()
            if self.active_layer_name == old_name: self.active_layer_name = new_name
            self._refresh_layer_list(); self._redraw_scene(); self._save_state()
    def delete_layer_dialog(self):
        if not self.project or not (sel := self.list_layers.currentItem()): return
        name = sel.text()
        for cell_name, cell in self.project.cells.items():
            if any(p.layer == name for p in cell.polygons) or any(e.layer == name for e in cell.ellipses):
                QMessageBox.warning(self, "Delete Error", f"Layer '{name}' is in use in cell '{cell_name}'."); return
        self.project.layers = [l for l in self.project.layers if l.name != name]; self.project.refresh_layer_map()
        if self.active_layer_name == name: self.active_layer_name = self.project.layers[0].name if self.project.layers else None
        self._refresh_layer_list(); self._save_state()

    def move_layer_forward(self): self._move_layer(1)
    def move_layer_backward(self): self._move_layer(-1)
    def move_layer_to_top(self):
        if self.project and (sel := self.list_layers.currentItem()) and (layer := self.project.layer_by_name.get(sel.text())):
            if self.project.layers.index(layer) < len(self.project.layers) - 1:
                self.project.layers.remove(layer); self.project.layers.append(layer)
                self._save_state(); self._refresh_layer_list(); self._redraw_scene()
    def move_layer_to_bottom(self):
        if self.project and (sel := self.list_layers.currentItem()) and (layer := self.project.layer_by_name.get(sel.text())):
            if self.project.layers.index(layer) > 0:
                self.project.layers.remove(layer); self.project.layers.insert(0, layer)
                self._save_state(); self._refresh_layer_list(); self._redraw_scene()
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
        if not self.project: return
        self.stop_vertex_edit(); self.scene.clear()
        if self.active_cell_name: self._draw_active_cell()
        if hasattr(self, 'view') and self.view:
            current_mode_name = next((name for name, val in self.view.MODES.items() if val == self.view.mode), "select")
            self.set_active_tool(current_mode_name)

    def _is_axis_aligned_rect(self, points):
        if len(points) != 4: return False
        xs, ys = set(p[0] for p in points), set(p[1] for p in points)
        return len(xs) == 2 and len(ys) == 2

    def _draw_active_cell(self):
        if not (cell := self.project.cells.get(self.active_cell_name)): return
        for layer in self.project.layers:
            if not layer.visible: continue
            for p_data in cell.polygons:
                if p_data.layer != layer.name: continue
                if self._is_axis_aligned_rect(p_data.points):
                    xs, ys = [p[0] for p in p_data.points], [p[1] for p in p_data.points]
                    self.scene.addItem(RectItem(QRectF(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)), layer, p_data))
                else:
                    self.scene.addItem(PolyItem(QPolygonF([QPointF(*pt) for pt in p_data.points]), layer, p_data))
            for e_data in cell.ellipses:
                if e_data.layer == layer.name: self.scene.addItem(CircleItem(QRectF(*e_data.rect), layer, e_data))
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
            QMenuBar { background-color: palette(window); }
            QMenuBar::item:selected { background: palette(highlight); }
        """)

    def rename_selected_shape(self):
        if not self.project: return
        selected_items = self.scene.selectedItems()
        if len(selected_items) != 1: self.statusBar().showMessage("Please select a single shape to rename."); return
        item = selected_items[0]
        if not hasattr(item, 'data_obj') or not hasattr(item.data_obj, 'name'):
            self.statusBar().showMessage("Selected item cannot be named."); return
        current_name = item.data_obj.name or ""
        new_name, ok = QInputDialog.getText(self, "Rename Shape", "Enter name:", text=current_name)
        if ok: item.data_obj.name = new_name if new_name else None; item.setToolTip(item.data_obj.name); self._save_state()

    def get_cell_bounds(self, cell_name):
        cell = self.project.cells.get(cell_name)
        if not cell: return QRectF()
        total_rect = QRectF()
        for p_data in cell.polygons:
            for pt in p_data.points: total_rect = total_rect.united(QRectF(QPointF(pt[0], pt[1]), QSize(0,0)))
        for e_data in cell.ellipses: total_rect = total_rect.united(QRectF(*e_data.rect))
        for ref in cell.references:
            child_bounds = self.get_cell_bounds(ref.cell)
            if not child_bounds.isNull():
                transform = QTransform().translate(ref.origin[0], ref.origin[1]).rotate(ref.rotation).scale(ref.magnification, ref.magnification)
                total_rect = total_rect.united(transform.mapRect(child_bounds))
        return total_rect

    def start_vertex_edit(self, poly_item):
        self.stop_vertex_edit()
        self._vertex_edit_item = poly_item
        poly_item.setFlag(QGraphicsItem.ItemIsMovable, False)
        for i in range(len(poly_item.polygon())):
            handle = VertexHandle(poly_item, i)
            self.scene.addItem(handle)
            poly_item.handles.append(handle)

    def stop_vertex_edit(self):
        if self._vertex_edit_item:
            self._vertex_edit_item.setFlag(QGraphicsItem.ItemIsMovable, True)
            for handle in self._vertex_edit_item.handles:
                self.scene.removeItem(handle)
            self._vertex_edit_item.handles.clear()
            self._vertex_edit_item = None

    def _run_boolean_op(self, op):
        if not self.project: return
        if not gdstk: QMessageBox.warning(self, "Feature Disabled", "Install 'gdstk'."); return
        
        # <<< MODIFIED: Include CircleItem in the selection
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
            
            # <<< ADDED: Handle CircleItem by converting it to a gdstk polygon (ellipse)
            elif isinstance(item, CircleItem):
                r = item.rect()
                center = (r.center().x(), r.center().y())
                radius = (r.width() / 2, r.height() / 2)
                # gdstk.ellipse returns a Polygon object, which is what we need
                gds_polys.append(gdstk.ellipse(center, radius))

        try:
            result_polys = gdstk.boolean(gds_polys[0], gds_polys[1:], op)
        except Exception as e:
            QMessageBox.critical(self, "Boolean Operation Failed", str(e)); return

        active_cell = self.project.cells[self.active_cell_name]
        data_to_delete = [item.data_obj for item in selected]
        
        # <<< MODIFIED: Ensure both polygons and ellipses (from circles) are removed
        uuids_to_delete = {d.uuid for d in data_to_delete}
        active_cell.polygons = [p for p in active_cell.polygons if p.uuid not in uuids_to_delete]
        active_cell.ellipses = [e for e in active_cell.ellipses if e.uuid not in uuids_to_delete]

        for res_poly in result_polys:
            active_cell.polygons.append(Poly(layer=first_layer, points=res_poly.points.tolist()))

        self._save_state()
        self._redraw_scene()


# -------------------------- main --------------------------
def main():
    app = QApplication(sys.argv)
    
    app.setOrganizationName("MyCompany")
    app.setApplicationName("2D Mask Layout Editor")

    # Fallback to current working directory if __file__ is not available
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

if __name__ == "__main__":
    main()
