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

from PyQt5.QtCore import Qt, QRectF, QPointF, QSize, QMimeData, pyqtSignal, QLineF, QUrl
from PyQt5.QtGui import (QBrush, QPen, QColor, QPolygonF, QPainter, QPixmap, QIcon, QPainterPath,
                         QDrag, QPalette, QFont, QTransform, QDesktopServices)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QColorDialog,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPolygonItem,
    QGraphicsItem, QToolBar, QPushButton, QLabel, QLineEdit,
    QSpinBox, QCheckBox, QDialog, QVBoxLayout, QDialogButtonBox, QToolTip,
    QDockWidget, QListWidget, QListWidgetItem, QWidget, QHBoxLayout, QMessageBox, QDoubleSpinBox, QGraphicsEllipseItem,
    QInputDialog, QTabWidget, QFrame, QToolButton, QGridLayout, QSlider, QGraphicsLineItem, QMenu, QGraphicsSceneMouseEvent
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

        if total_height > 0:
            w, h, d = max_x - min_x, max_y - min_y, total_height
            v = [
                (-w/2, -h/2, 0), (w/2, -h/2, 0), (w/2, h/2, 0), (-w/2, h/2, 0),
                (-w/2, -h/2, d), (w/2, -h/2, d), (w/2, h/2, d), (-w/2, h/2, d)
            ]
            box_faces_indices = [
                (0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 7, 4),
                (1, 2, 6, 5), (0, 1, 5, 4), (2, 3, 7, 6)
            ]
            box_color = QColor(150, 150, 150, 40)
            for face_indices in box_faces_indices:
                all_faces.append(([v[i] for i in face_indices], box_color))

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

class SceneItemMixin:
    def __init__(self, *args, **kwargs): pass
    def customMousePressEvent(self, event): pass
    def customMouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsItem.ItemIsMovable:
            if event.scenePos() != event.buttonDownScenePos(Qt.LeftButton):
                self.scene().views()[0].parent_win.update_data_from_item_move(self)

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
        color = self.base_color.lighter(130) if selected else self.base_color
        color.setAlpha(180)
        self.setBrush(QBrush(color)); self.setPen(QPen(Qt.black, 0.5))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange: self.refresh_appearance(selected=bool(value))
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if event is None: event = QGraphicsSceneMouseEvent()
        super().mouseDoubleClickEvent(event)
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
        color = self.base_color.lighter(130) if selected else self.base_color
        color.setAlpha(180)
        self.setBrush(QBrush(color)); self.setPen(QPen(Qt.black, 0.5))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange: self.refresh_appearance(selected=bool(value))
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if event is None: event = QGraphicsSceneMouseEvent()
        super().mouseDoubleClickEvent(event)
        self.scene().views()[0].parent_win.show_properties_dialog_for_item(self)

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
        color = self.base_color.lighter(130) if selected else self.base_color
        color.setAlpha(180); self.setBrush(QBrush(color)); self.setPen(QPen(Qt.black, 0.5))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange: self.refresh_appearance(selected=bool(value))
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if event is None: event = QGraphicsSceneMouseEvent()
        super().mouseDoubleClickEvent(event)
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

    def mousePressEvent(self, event): super().mousePressEvent(event); self._drag_start_pos = self.pos()
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsItem.ItemIsMovable and self.pos() != self._drag_start_pos:
            self.scene().views()[0].parent_win.update_data_from_item_move(self)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            if self.scene().views()[0].parent_win.chk_snap.isChecked():
                return self.scene().views()[0].snap_point(value)
        return super().itemChange(change, value)

    def boundingRect(self): return self.childrenBoundingRect() or QRectF(-5, -5, 10, 10)
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
            # Package the cell name as plain text
            mimeData.setText(item.text())
            
            drag = QDrag(self)
            drag.setMimeData(mimeData)
            
            # Start the drag using the allowed actions
            drag.exec_(Qt.CopyAction | Qt.LinkAction)

# -------------------------- View/Scene with Rulers --------------------------

class Ruler(QWidget):
    def __init__(self, orientation, canvas, parent=None):
        super().__init__(parent)
        self.canvas, self.orientation = canvas, orientation
        self.ruler_font = QFont("Arial", 8)
        self.setFixedHeight(30) if orientation == Qt.Horizontal else self.setFixedWidth(30)

    def paintEvent(self, event):
        if not self.canvas.parent_win.project: return
        painter = QPainter(self); painter.setFont(self.ruler_font)
        bg, tick = self.palette().color(QPalette.Window), self.palette().color(QPalette.WindowText)
        painter.fillRect(self.rect(), bg)
        visible = self.canvas.mapToScene(self.canvas.viewport().rect()).boundingRect()
        zoom = self.canvas.transform().m11()
        pitch = float(self.canvas.parent_win.project.grid_pitch)
        if pitch <= 0: return
        req_pitch = 8
        while (pitch * zoom) < req_pitch: pitch *= 5
        while (pitch * zoom) > req_pitch * 5: pitch /= 5
        painter.setPen(QPen(tick, 1))
        if self.orientation == Qt.Horizontal:
            start = math.floor(visible.left() / pitch) * pitch
            end = math.ceil(visible.right() / pitch) * pitch
            count = int(round(start / pitch))
            x = start
            while x <= end:
                vx = self.canvas.mapFromScene(QPointF(x, 0)).x()
                h = 10 if count % 5 == 0 else 5
                painter.drawLine(vx, self.height(), vx, self.height() - h)
                if count % 5 == 0 and zoom > 0.05:
                    painter.drawText(vx + 2, self.height() - 12, f"{x:.0f}")
                x += pitch; count += 1
        else:
            start = math.floor(visible.top() / pitch) * pitch
            end = math.ceil(visible.bottom() / pitch) * pitch
            count = int(round(start / pitch))
            y = start
            while y <= end:
                vy = self.canvas.mapFromScene(QPointF(0, y)).y()
                w = 10 if count % 5 == 0 else 5
                painter.drawLine(self.width(), vy, self.width() - w, vy)
                if count % 5 == 0 and zoom > 0.05:
                    painter.save()
                    painter.translate(self.width() - 12, vy - 2); painter.rotate(90)
                    painter.drawText(0, 0, f"{-y:.0f}")
                    painter.restore()
                y += pitch; count += 1

class CanvasContainer(QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.h_ruler, self.v_ruler = Ruler(Qt.Horizontal, self.canvas), Ruler(Qt.Vertical, self.canvas)
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
    MODES = {"select": 1, "rect": 2, "poly": 3, "circle": 4, "move": 5, "measure": 6, "path": 7}
    zoomChanged = pyqtSignal()

    def __init__(self, scene, get_active_layer, parent=None):
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing); self.setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse); self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.mode = self.MODES["select"]; self.start_pos = None; self.temp_item = None
        self.temp_poly_points: List[QPointF] = []
        self.temp_path_points: List[QPointF] = []
        self.get_active_layer, self.parent_win = get_active_layer, parent
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._panning = False

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
        req_pitch = 8
        while (pitch * zoom) < req_pitch: pitch *= 5
        while (pitch * zoom) > req_pitch * 5: pitch /= 5
        minor_pen, major_pen = QPen(minor_c, 0), QPen(major_c, 0)
        left, right, top, bottom = visible.left(), visible.right(), visible.top(), visible.bottom()
        x_start, x_end = math.floor(left / pitch) * pitch, math.ceil(right / pitch) * pitch
        count = int(round(x_start / pitch)); x = x_start
        while x <= x_end:
            painter.setPen(major_pen if count % 5 == 0 else minor_pen)
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            x += pitch; count += 1
        y_start, y_end = math.floor(top / pitch) * pitch, math.ceil(bottom / pitch) * pitch
        count = int(round(y_start / pitch)); y = y_start
        while y <= y_end:
            painter.setPen(major_pen if count % 5 == 0 else minor_pen)
            painter.drawLine(QPointF(left, y), QPointF(right, y))
            y += pitch; count += 1

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
                elif self.temp_poly_points:
                    marker_size = 2
                    marker_rect = QRectF(scene_pos.x() - marker_size / 2, scene_pos.y() - marker_size / 2, marker_size, marker_size)
                    self.temp_item = self.scene().addRect(marker_rect, QPen(Qt.gray, 0))
            elif self.mode == self.MODES["path"]:
                self.temp_path_points.append(scene_pos)
                if self.temp_item: self.scene().removeItem(self.temp_item)
                if len(self.temp_path_points) > 1:
                    poly = QPolygonF(self.temp_path_points)
                    path = QPainterPath(); path.addPolygon(poly)
                    self.temp_item = self.scene().addPath(path, QPen(Qt.gray, 0, Qt.DashLine))
            else:
                super().mousePressEvent(event)
                return
            event.accept()
        else:
            super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        menu = QMenu(self)
        
        paste_action = menu.addAction("Paste")
        paste_action.triggered.connect(lambda: self.parent_win.paste_shapes(self.mapToScene(event.pos())))
        if not self.parent_win.can_paste(): paste_action.setEnabled(False)
        
        if isinstance(item, (RectItem, PolyItem, CircleItem, RefItem)):
            if not item.isSelected():
                self.scene().clearSelection()
                item.setSelected(True)
            
            menu.addSeparator()
            prop = menu.addAction("Properties..."); prop.triggered.connect(lambda: self.parent_win.show_properties_dialog_for_item(item))
            rename = menu.addAction("Rename..."); rename.triggered.connect(self.parent_win.rename_selected_shape)
            measure = menu.addAction("Measure..."); measure.triggered.connect(self.parent_win.measure_selection)
            
            menu.addSeparator()
            copy = menu.addAction("Copy"); copy.triggered.connect(self.parent_win.copy_selected_shapes)
            
            # Arrange Submenu
            arrange_menu = menu.addMenu("Arrange")
            front = arrange_menu.addAction("Bring to Front")
            front.triggered.connect(self.parent_win.bring_selection_to_front)
            forward = arrange_menu.addAction("Move Forward")
            forward.triggered.connect(self.parent_win.move_selection_forward)
            backward = arrange_menu.addAction("Move Backward")
            backward.triggered.connect(self.parent_win.move_selection_backward)
            back = arrange_menu.addAction("Send to Back")
            back.triggered.connect(self.parent_win.send_selection_to_back)

            # Transform Submenu
            transform_menu = menu.addMenu("Transform")
            rot = transform_menu.addAction("Rotate..."); rot.triggered.connect(self.parent_win.rotate_selection)
            sca = transform_menu.addAction("Scale..."); sca.triggered.connect(self.parent_win.scale_selection)
            flip_menu = transform_menu.addMenu("Flip")
            fh = flip_menu.addAction("Horizontal"); fh.triggered.connect(self.parent_win.flip_selection_horizontal)
            fv = flip_menu.addAction("Vertical"); fv.triggered.connect(self.parent_win.flip_selection_vertical)
            
            # Move to Layer Submenu
            move_menu = menu.addMenu("Move to Layer")
            for layer in self.parent_win.project.layers:
                layer_action = move_menu.addAction(layer.name)
                layer_action.triggered.connect(lambda checked=False, l=layer.name: self.parent_win.move_selection_to_layer(l))
            
            menu.addSeparator()
            delete = menu.addAction("Delete"); delete.triggered.connect(self.parent_win.delete_selected_items)
            
        menu.exec_(event.globalPos())

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
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
                line = QLineF(self.start_pos, snapped_pos)
                self.temp_item.setLine(line)
                self.parent_win.statusBar().showMessage(f"dX: {line.dx():.2f}, dY: {-line.dy():.2f}, Dist: {line.length():.2f}")
            event.accept(); return
        elif self.mode == self.MODES["poly"] and len(self.temp_poly_points) > 0:
            if self.temp_item: self.scene().removeItem(self.temp_item)
            poly_pts = self.temp_poly_points + [self.snap_point(scene_pos)]
            poly = QPolygonF(poly_pts)
            self.temp_item = self.scene().addPolygon(poly, QPen(Qt.gray, 0, Qt.DashLine))
        elif self.mode == self.MODES["path"] and len(self.temp_path_points) > 0:
            if self.temp_item: self.scene().removeItem(self.temp_item)
            path_pts = self.temp_path_points + [self.snap_point(scene_pos)]
            poly = QPolygonF(path_pts)
            path = QPainterPath(); path.addPolygon(poly)
            self.temp_item = self.scene().addPath(path, QPen(Qt.gray, 0, Qt.DashLine))

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False; self.setCursor(Qt.ArrowCursor); event.accept(); return
        if self.start_pos and self.mode in [self.MODES["rect"], self.MODES["circle"]]:
            rect = self.temp_item.rect()
            self.temp_cancel()
            if rect.width() > 0 and rect.height() > 0 and (layer := self.get_active_layer()):
                if self.mode == self.MODES["rect"]: self.parent_win.add_rect_to_active_cell(rect, layer)
                else: self.parent_win.add_circle_to_active_cell(rect, layer)
            event.accept(); return
        if self.start_pos and self.mode == self.MODES["measure"]:
            self.temp_cancel(); event.accept(); return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.mode == self.MODES["path"] and event.button() == Qt.LeftButton:
            self.finish_path(); event.accept(); return
        if self.mode == self.MODES["poly"] and event.button() == Qt.LeftButton:
            self.finish_polygon(); event.accept(); return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.parent_win.set_active_tool("select")
            self.temp_cancel()
        else: 
            super().keyPressEvent(event)

    def finish_polygon(self):
        can_create = len(self.temp_poly_points) >= 3 and (layer := self.get_active_layer())
        if can_create: poly = QPolygonF(self.temp_poly_points)
        self.temp_cancel()
        if can_create: self.parent_win.add_poly_to_active_cell(poly, layer)
    
    def finish_path(self):
        can_create = len(self.temp_path_points) >= 2 and (layer := self.get_active_layer())
        if can_create:
            points = self.temp_path_points[:] # Make a copy
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

        if isinstance(item_at_pos, (RectItem, PolyItem, CircleItem)) and \
           (item_at_pos.flags() & QGraphicsItem.ItemIsSelectable):
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
            filters = "Layout Files (*.json *.gds *.oas);;All Files (*)"
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
        project = Project(
            units=data.get('units', 'um'), top=data.get('top', 'TOP'),
            layers=[Layer(**l) for l in data.get('layers', [])],
            grid_pitch=data.get('grid_pitch', 50),
            canvas_width=data.get('canvas_width', 2000), canvas_height=data.get('canvas_height', 1500)
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
                    layer_meta = {int(k): v for k, v in json.load(f).get('layer_metadata', {}).items()}
            except Exception as e: print(f"Warning: Could not read sidecar metadata file. {e}")
        project = self._create_default_project(50, 2000, 1500)
        project.layers.clear(); project.cells.clear()
        gds_layers = sorted(list(lib.layers))
        for l_num in gds_layers:
            if l_num in layer_meta:
                meta = layer_meta[l_num]
                project.layers.append(Layer(name=meta['name'], color=tuple(meta['color']), visible=meta['visible']))
            else:
                project.layers.append(Layer(f"Layer_{l_num}", (l_num*20%255, l_num*50%255, l_num*80%255)))
        project.refresh_layer_map()
        layer_num_to_name = {l_num: next(l.name for l in project.layers if l.name.endswith(f"_{l_num}")) for l_num in gds_layers}
        for cell in lib.cells:
            new_cell = Cell()
            for poly in cell.polygons:
                new_cell.polygons.append(Poly(layer_num_to_name.get(poly.layer, f"Layer_{poly.layer}"), poly.points.tolist()))
            for ref in cell.references:
                ref_cell_name = ref.cell.name if isinstance(ref.cell, gdstk.Cell) else ref.cell
                new_cell.references.append(Ref(ref_cell_name, ref.origin, ref.rotation, ref.magnification))
            project.cells[cell.name] = new_cell
        if lib.top_level(): project.top = lib.top_level()[0].name
        elif project.cells: project.top = list(project.cells.keys())[0]
        self.current_file_path = None
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
            lib = gdstk.Library(unit=float(self.project.units.replace('um','e-6')))
            layer_map = {layer.name: i for i, layer in enumerate(self.project.layers, 1)}
            gds_cells = {name: lib.new_cell(name) for name in self.project.cells}
            for name, cell in self.project.cells.items():
                gds_cell = gds_cells[name]
                for p in cell.polygons: gds_cell.add(gdstk.Polygon(p.points, layer=layer_map.get(p.layer, 0)))
                for e in cell.ellipses:
                    r = e.rect; center = (r[0]+r[2]/2, r[1]+r[3]/2); radius = (r[2]/2)
                    gds_cell.add(gdstk.ellipse(center, radius, layer=layer_map.get(e.layer, 0)))
                for r in cell.references:
                    if r.cell in gds_cells:
                        gds_cell.add(gdstk.Reference(gds_cells[r.cell], r.origin, r.rotation, r.magnification))
            lib.write_oas(path) if is_oas else lib.write_gds(path)
            meta = {layer_map[l.name]: {'name': l.name, 'color': l.color, 'visible': l.visible} for l in self.project.layers if l.name in layer_map}
            with open(path + '.json', 'w') as f: json.dump({'layer_metadata': meta}, f, indent=2)
            self.statusBar().showMessage(f"Exported to {path} and created metadata file.")
        except Exception as e: QMessageBox.critical(self, "Export Error", f"Failed to write file: {e}")
        finally: QApplication.restoreOverrideCursor()

    # --- UI Building & Core Logic ---

    def _create_themed_icon(self, icon_name):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
            icon_path = os.path.join(script_dir, 'icons', f'{icon_name}.svg')
            pixmap = QPixmap(icon_path)
            if self.palette().color(QPalette.Window).value() < 128:
                painter = QPainter(pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                painter.fillRect(pixmap.rect(), Qt.white); painter.end()
            return QIcon(pixmap)
        except Exception:
            pixmap = QPixmap(32, 32); pixmap.fill(Qt.transparent)
            p = QPainter(pixmap); p.setPen(QPen(Qt.gray, 2)); p.drawText(pixmap.rect(), Qt.AlignCenter, icon_name[0].upper()); p.end()
            return QIcon(pixmap)

    def _build_ui(self):
        self.tool_buttons = {}
        self._create_shortcuts()
        self._build_tool_tabs()
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

    def _build_tool_tabs(self):
        dock = QDockWidget(self); dock.setTitleBarWidget(QWidget()); dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetMovable); tabs = QTabWidget(); dock.setWidget(tabs)

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
        main_layout.addStretch(); tabs.addTab(main_tab, "Main")

        draw_tab = QWidget(); draw_layout = QHBoxLayout(draw_tab)
        draw_layout.setAlignment(Qt.AlignLeft); draw_layout.setContentsMargins(0, 5, 0, 5)
        draw_layout.addWidget(self._create_ribbon_group("Create", [
            self._create_tool_button("rect", "Rectangle", "square"), 
            self._create_tool_button("circle", "Circle", "circle"), 
            self._create_tool_button("poly", "Polygon", "pen-tool"),
            self._create_action_button("Finish Poly", "check-square", lambda: self.view.finish_polygon()),
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
        draw_layout.addStretch(); tabs.addTab(draw_tab, "Draw")

        view_tab = QWidget(); view_layout = QHBoxLayout(view_tab)
        view_layout.setAlignment(Qt.AlignLeft); view_layout.setContentsMargins(0, 5, 0, 5)
        toggle_rulers_btn = self._create_action_button("Rulers", "table", lambda checked: self.toggle_rulers(checked))
        toggle_rulers_btn.setCheckable(True); toggle_rulers_btn.setChecked(True)
        view_layout.addWidget(self._create_ribbon_group("Display", [
            self._create_action_button("Fill", "maximize", self.fill_view),
            self._create_action_button("3D View", "box", self.show_3d_view),
            toggle_rulers_btn, self._create_tool_button("measure", "Measure", "compass")
        ]))
        view_layout.addWidget(self._create_ribbon_group("Zoom", [
            self._create_action_button("Zoom In", "zoom-in", self.zoom_in), self._create_action_button("Zoom Out", "zoom-out", self.zoom_out)
        ]))
        view_layout.addWidget(self._create_ribbon_group("Layer Visibility", [
            self._create_action_button("Show All", "eye", self.show_all_layers), self._create_action_button("Hide All", "eye-off", self.hide_all_layers)
        ]))
        view_layout.addStretch(); tabs.addTab(view_tab, "View")
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
        self.move_layer_menu.clear()
        for layer in self.project.layers:
            item = QListWidgetItem(layer.name); item.setIcon(self._create_color_icon(QColor(*layer.color)))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable); item.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
            self.list_layers.addItem(item)
            if layer.name == current: self.list_layers.setCurrentItem(item)
            action = self.move_layer_menu.addAction(layer.name)
            action.triggered.connect(lambda checked=False, l=layer.name: self.move_selection_to_layer(l))
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
        self._undo_stack.append(copy.deepcopy(self.project))
        if len(self._undo_stack) > 50: self._undo_stack.pop(0)
        if len(self._undo_stack) > 1: self._mark_dirty()

    def undo(self):
        if len(self._undo_stack) > 1:
            self._redo_stack.append(self._undo_stack.pop())
            self.project = copy.deepcopy(self._undo_stack[-1])
            self.update_ui_from_project(); self._mark_dirty()
    def redo(self):
        if self._redo_stack:
            self.project = self._redo_stack.pop()
            self._undo_stack.append(copy.deepcopy(self.project))
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

        width, ok = QInputDialog.getDouble(self, "Path Width", "Enter path width:", 10.0, 0.01, 10000.0, 2)
        if not ok: return
        
        try:
            point_list = [(p.x(), p.y()) for p in points]
            # Use gdstk to convert the centerline and width into a polygon
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
        bounds = item.mapToScene(item.boundingRect()).boundingRect()
        delta = self.view.snap_point(bounds.topLeft()) - bounds.topLeft()
        if delta.x() == 0 and delta.y() == 0: self._save_state(); return

        if isinstance(item, RefItem):
            pos = item.pos() + delta
            item.ref.origin = (pos.x(), pos.y())
        elif isinstance(item, CircleItem):
            rect = item.mapToScene(item.rect()).boundingRect(); rect.translate(delta)
            item.data_obj.rect = (rect.x(), rect.y(), rect.width(), rect.height())
        elif isinstance(item, (PolyItem, RectItem)):
            poly = item.polygon() if isinstance(item, PolyItem) else QPolygonF(item.rect())
            poly = item.mapToScene(poly)
            item.data_obj.points = [(p.x() + delta.x(), p.y() + delta.y()) for p in poly]
        self._save_state(); self._redraw_scene()

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
        
        to_delete = {item.data_obj for item in selected if hasattr(item, 'data_obj')}
        to_delete.update({item.ref for item in selected if isinstance(item, RefItem)})
        
        if to_delete:
            self.scene.clearSelection()
            cell.polygons = [p for p in cell.polygons if p not in to_delete]
            cell.ellipses = [e for e in cell.ellipses if e not in to_delete]
            cell.references = [r for r in cell.references if r not in to_delete]
            self._redraw_scene(); self._save_state()

    def fillet_selected_poly(self):
        if not gdstk: QMessageBox.warning(self, "Feature Disabled", "Please install 'gdstk'."); return
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

    def fill_view(self):
        if not (self.project and self.scene.items()): return
        rect = self.scene.itemsBoundingRect()
        if not rect.isNull():
            margin = max(rect.width(), rect.height()) * 0.05
            self.view.fitInView(rect.adjusted(-margin, -margin, margin, margin), Qt.KeepAspectRatio)
            self.view.zoomChanged.emit()

    def show_3d_view(self):
        if self.project: ThreeDViewDialog(self.project, self).exec_()

    def export_svg(self):
        if not self.project: return
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", "", "SVG Files (*.svg)")
        if not path: return
        gen = QSvgGenerator(); gen.setFileName(path)
        rect = self.scene.itemsBoundingRect().adjusted(-50, -50, 50, 50)
        gen.setViewBox(rect)
        painter = QPainter(gen)
        self.scene.render(painter, target=QRectF(), source=rect)
        painter.end()

    def set_grid_pitch(self, value):
        if self.project: self.project.grid_pitch = value; self.view.viewport().update(); self._mark_dirty()

    def resnap_all_items_to_grid(self):
        if not self.project: return
        reply = QMessageBox.question(self, 'Re-snap Cell', "This will snap all shapes in the current cell to the grid. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No or self.project.grid_pitch <= 0: return
        if not (active_cell := self.project.cells.get(self.active_cell_name)): return
        def snap(v): return round(v / self.project.grid_pitch) * self.project.grid_pitch
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
        self.scene.clear()
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
        is_first = True

        def unite_rect(rect_to_add):
            nonlocal total_rect, is_first
            if is_first and not rect_to_add.isNull():
                total_rect = QRectF(rect_to_add) # Start with the first valid rect
                is_first = False
            else:
                total_rect = total_rect.united(rect_to_add)

        for p_data in cell.polygons:
            if not p_data.points: continue
            min_x = min(p[0] for p in p_data.points)
            max_x = max(p[0] for p in p_data.points)
            min_y = min(p[1] for p in p_data.points)
            max_y = max(p[1] for p in p_data.points)
            unite_rect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))
                
        for e_data in cell.ellipses:
            unite_rect(QRectF(*e_data.rect))
        
        for ref in cell.references:
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
                gds_polys.append(gdstk.ellipse(center, radius))
        try:
            result_polys = gdstk.boolean(gds_polys[0], gds_polys[1:], op)
        except Exception as e:
            QMessageBox.critical(self, "Boolean Operation Failed", str(e)); return
        active_cell = self.project.cells[self.active_cell_name]
        data_to_delete = [item.data_obj for item in selected]
        uuids_to_delete = {d.uuid for d in data_to_delete}
        active_cell.polygons = [p for p in active_cell.polygons if p.uuid not in uuids_to_delete]
        active_cell.ellipses = [e for e in active_cell.ellipses if e.uuid not in uuids_to_delete]
        for res_poly in result_polys:
            active_cell.polygons.append(Poly(layer=first_layer, points=res_poly.points.tolist()))
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
            for item_data in pasted_items:
                if item_data.get('type') == 'poly':
                    for x, y in item_data['points']:
                        min_x, max_x = min(min_x, x), max(max_x, x)
                        min_y, max_y = min(min_y, y), max(max_y, y)
                elif item_data.get('type') == 'ellipse':
                    x, y, w, h = item_data['rect']
                    min_x, max_x = min(min_x, x), max(max_x, x + w)
                    min_y, max_y = min(min_y, y), max(max_y, y + h)

            center_x = (min_x + max_x) / 2 if min_x != float('inf') else 0
            center_y = (min_y + max_y) / 2 if min_y != float('inf') else 0
            
            target_pos = self.view.snap_point(scene_pos) if scene_pos else QPointF(center_x + offset, center_y + offset)
            delta = target_pos - QPointF(center_x, center_y)

            newly_pasted_data = []
            for item_data in pasted_items:
                item_data.pop('uuid', None)
                item_type = item_data.pop('type', None)
                if item_type == 'poly':
                    item_data['points'] = [(p[0] + delta.x(), p[1] + delta.y()) for p in item_data['points']]
                    new_shape = Poly(**item_data)
                    active_cell.polygons.append(new_shape)
                    newly_pasted_data.append(new_shape)
                elif item_type == 'ellipse':
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
                item.setSelected(True)

    def zoom_in(self):
        if hasattr(self, 'view'): self.view.scale(1.2, 1.2); self.view.zoomChanged.emit()
    def zoom_out(self):
        if hasattr(self, 'view'): self.view.scale(1 / 1.2, 1 / 1.2); self.view.zoomChanged.emit()
    def show_all_layers(self):
        if not self.project: return
        for layer in self.project.layers: layer.visible = True
        self._refresh_layer_list(); self._redraw_scene(); self._save_state()
    def hide_all_layers(self):
        if not self.project: return
        for layer in self.project.layers: layer.visible = False
        self._refresh_layer_list(); self._redraw_scene(); self._save_state()
    def toggle_rulers(self, checked):
        if hasattr(self, 'canvas_container'):
            self.canvas_container.h_ruler.setVisible(checked)
            self.canvas_container.v_ruler.setVisible(checked)

    def show_properties_dialog_for_item(self, item):
        """Opens the correct properties dialog for the given graphics item."""
        if not item or not hasattr(item, 'data_obj'): return
        if isinstance(item, (RectItem, PolyItem)):
            if isinstance(item, RectItem):
                r = item.rect()
                kind, params = "rect", {"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height()}
            else:
                pts = [(p.x(), p.y()) for p in item.polygon()]
                kind, params = "poly", {"points": pts}
            dlg = ShapeEditDialog(kind, params, self.view)
            if dlg.exec_() == QDialog.Accepted and (vals := dlg.get_values()):
                if kind == "rect": item.setRect(QRectF(vals["x"], vals["y"], vals["w"], vals["h"]))
                else: item.setPolygon(QPolygonF([QPointF(x, y) for x, y in vals["points"]]))
                self.update_data_from_item_edit(item)
        elif isinstance(item, CircleItem):
            r = item.rect()
            params = {"center_x": r.center().x(), "center_y": r.center().y(), "w": r.width(), "h": r.height()}
            dlg = ShapeEditDialog("circle", params, self.view)
            if dlg.exec_() == QDialog.Accepted and (vals := dlg.get_values()):
                cx, cy, w, h = vals["center_x"], vals["center_y"], vals["w"], vals["h"]
                item.setRect(QRectF(cx - w / 2, cy - h / 2, w, h))
                self.update_data_from_item_edit(item)

    def can_paste(self):
        """Checks if the clipboard contains pastable shape data."""
        try:
            data = json.loads(QApplication.clipboard().text())
            return "layout_editor_clipboard" in data
        except (json.JSONDecodeError, TypeError):
            return False

    def get_selection_bounds_and_center(self):
        """Calculates the bounding box and center of the current selection."""
        if not self.scene.selectedItems(): return QRectF(), QPointF()
        bounds = QRectF()
        for item in self.scene.selectedItems():
            bounds = bounds.united(item.sceneBoundingRect())
        return bounds, bounds.center()

    def _transform_shapes(self, transform_func):
        """Generic handler for transformations (rotate, scale, flip)."""
        if not self.project or not self.scene.selectedItems(): return
        
        active_cell = self.project.cells[self.active_cell_name]
        items_to_transform = self.scene.selectedItems()
        
        # Convert any selected ellipses to polygons for robust transformation
        new_polys_from_ellipses = []
        for item in items_to_transform:
            if isinstance(item, CircleItem):
                ellipse_obj = item.data_obj
                r = item.rect()
                # Use gdstk for a good circular approximation
                gds_ellipse = gdstk.ellipse((r.center().x(), r.center().y()), (r.width() / 2, r.height() / 2))
                new_poly = Poly(layer=ellipse_obj.layer, name=ellipse_obj.name, points=gds_ellipse.points.tolist())
                new_polys_from_ellipses.append(new_poly)
                # Remove old ellipse
                active_cell.ellipses = [e for e in active_cell.ellipses if e.uuid != ellipse_obj.uuid]
        
        if new_polys_from_ellipses:
            active_cell.polygons.extend(new_polys_from_ellipses)
            # Redraw scene to get new PolyItems, then re-select them for transformation
            self._redraw_scene()
            all_items = self.scene.items()
            for item in all_items:
                if hasattr(item, 'data_obj') and item.data_obj in new_polys_from_ellipses:
                    item.setSelected(True)
        
        transform_func() # Apply the actual transformation
        self._save_state()
        self._redraw_scene()

    def rotate_selection(self):
        angle, ok = QInputDialog.getDouble(self, "Rotate Selection", "Angle (degrees):", 0, -360, 360, 1)
        if not ok: return
        
        def do_rotate():
            _, center = self.get_selection_bounds_and_center()
            transform = QTransform().translate(center.x(), center.y()).rotate(angle).translate(-center.x(), -center.y())
            
            for item in self.scene.selectedItems():
                if hasattr(item, 'data_obj') and isinstance(item.data_obj, Poly):
                    # Get the shape as a QPolygonF regardless of item type
                    if isinstance(item, PolyItem):
                        poly = item.polygon()
                    elif isinstance(item, RectItem):
                        poly = QPolygonF(item.rect())
                    else:
                        continue # Skip other item types
                        
                    new_poly = transform.map(poly)
                    item.data_obj.points = [(p.x(), p.y()) for p in new_poly]
        
        self._transform_shapes(do_rotate)

    def scale_selection(self):
        factor, ok = QInputDialog.getDouble(self, "Scale Selection", "Scale Factor:", 1.0, 0.01, 100.0, 2)
        if not ok or factor == 1.0: return

        def do_scale():
            _, center = self.get_selection_bounds_and_center()
            transform = QTransform().translate(center.x(), center.y()).scale(factor, factor).translate(-center.x(), -center.y())
            
            for item in self.scene.selectedItems():
                if hasattr(item, 'data_obj') and isinstance(item.data_obj, Poly):
                    # Get the shape as a QPolygonF regardless of item type
                    if isinstance(item, PolyItem):
                        poly = item.polygon()
                    elif isinstance(item, RectItem):
                        poly = QPolygonF(item.rect())
                    else:
                        continue # Skip other item types

                    new_poly = transform.map(poly)
                    item.data_obj.points = [(p.x(), p.y()) for p in new_poly]
        
        self._transform_shapes(do_scale)

    def flip_selection_horizontal(self):
        def do_flip():
            _, center = self.get_selection_bounds_and_center()
            transform = QTransform().translate(center.x(), 0).scale(-1, 1).translate(-center.x(), 0)
            
            for item in self.scene.selectedItems():
                if hasattr(item, 'data_obj') and isinstance(item.data_obj, Poly):
                    # Get the shape as a QPolygonF regardless of item type
                    if isinstance(item, PolyItem):
                        poly = item.polygon()
                    elif isinstance(item, RectItem):
                        poly = QPolygonF(item.rect())
                    else:
                        continue # Skip other item types
                        
                    new_poly = transform.map(poly)
                    item.data_obj.points = [(p.x(), p.y()) for p in new_poly]
        self._transform_shapes(do_flip)
        
    def flip_selection_vertical(self):
        def do_flip():
            _, center = self.get_selection_bounds_and_center()
            transform = QTransform().translate(0, center.y()).scale(1, -1).translate(0, -center.y())
            
            for item in self.scene.selectedItems():
                if hasattr(item, 'data_obj') and isinstance(item.data_obj, Poly):
                    # Get the shape as a QPolygonF regardless of item type
                    if isinstance(item, PolyItem):
                        poly = item.polygon()
                    elif isinstance(item, RectItem):
                        poly = QPolygonF(item.rect())
                    else:
                        continue # Skip other item types
                        
                    new_poly = transform.map(poly)
                    item.data_obj.points = [(p.x(), p.y()) for p in new_poly]
        self._transform_shapes(do_flip)

    def flip_selection(self):
        menu = QMenu(self)
        horiz = menu.addAction("Flip Horizontal")
        horiz.triggered.connect(self.flip_selection_horizontal)
        vert = menu.addAction("Flip Vertical")
        vert.triggered.connect(self.flip_selection_vertical)
        menu.exec_(self.cursor().pos())

    def arrange_selection(self):
        menu = QMenu(self)
        front = menu.addAction("Bring to Front")
        front.triggered.connect(self.bring_selection_to_front)
        back = menu.addAction("Send to Back")
        back.triggered.connect(self.send_selection_to_back)
        menu.exec_(self.cursor().pos())

    def move_selection_forward(self):
        if not (cell := self.project.cells.get(self.active_cell_name)) or not self.scene.selectedItems(): return
        selected_data = {item.data_obj for item in self.scene.selectedItems() if hasattr(item, 'data_obj')}
        
        # Move polygons forward
        for i in range(len(cell.polygons) - 2, -1, -1):
            if cell.polygons[i] in selected_data and cell.polygons[i+1] not in selected_data:
                cell.polygons[i], cell.polygons[i+1] = cell.polygons[i+1], cell.polygons[i]

        # Move ellipses forward
        for i in range(len(cell.ellipses) - 2, -1, -1):
            if cell.ellipses[i] in selected_data and cell.ellipses[i+1] not in selected_data:
                cell.ellipses[i], cell.ellipses[i+1] = cell.ellipses[i+1], cell.ellipses[i]

        self._save_state(); self._redraw_scene()

    def move_selection_backward(self):
        if not (cell := self.project.cells.get(self.active_cell_name)) or not self.scene.selectedItems(): return
        selected_data = {item.data_obj for item in self.scene.selectedItems() if hasattr(item, 'data_obj')}

        # Move polygons backward
        for i in range(1, len(cell.polygons)):
            if cell.polygons[i] in selected_data and cell.polygons[i-1] not in selected_data:
                cell.polygons[i], cell.polygons[i-1] = cell.polygons[i-1], cell.polygons[i]
        
        # Move ellipses backward
        for i in range(1, len(cell.ellipses)):
            if cell.ellipses[i] in selected_data and cell.ellipses[i-1] not in selected_data:
                cell.ellipses[i], cell.ellipses[i-1] = cell.ellipses[i-1], cell.ellipses[i]

        self._save_state(); self._redraw_scene()

    def bring_selection_to_front(self):
        if not (cell := self.project.cells.get(self.active_cell_name)) or not self.scene.selectedItems(): return
        selected_data = {item.data_obj for item in self.scene.selectedItems() if hasattr(item, 'data_obj')}
        
        polys_to_move = [p for p in cell.polygons if p in selected_data]
        ellipses_to_move = [e for e in cell.ellipses if e in selected_data]
        
        cell.polygons = [p for p in cell.polygons if p not in selected_data] + polys_to_move
        cell.ellipses = [e for e in cell.ellipses if e not in selected_data] + ellipses_to_move
        
        self._save_state(); self._redraw_scene()

    def send_selection_to_back(self):
        if not (cell := self.project.cells.get(self.active_cell_name)) or not self.scene.selectedItems(): return
        selected_data = {item.data_obj for item in self.scene.selectedItems() if hasattr(item, 'data_obj')}
        
        polys_to_move = [p for p in cell.polygons if p in selected_data]
        ellipses_to_move = [e for e in cell.ellipses if e in selected_data]

        cell.polygons = polys_to_move + [p for p in cell.polygons if p not in selected_data]
        cell.ellipses = ellipses_to_move + [e for e in cell.ellipses if e not in selected_data]

        self._save_state(); self._redraw_scene()
        
    def move_selection_to_layer(self, layer_name):
        if not self.scene.selectedItems(): return
        for item in self.scene.selectedItems():
            if hasattr(item, 'data_obj'):
                item.data_obj.layer = layer_name
        self._save_state(); self._redraw_scene()

    def measure_selection(self):
        if not self.scene.selectedItems() or not gdstk: return
        
        total_area, total_perimeter = 0.0, 0.0
        
        for item in self.scene.selectedItems():
            try:
                if isinstance(item, PolyItem):
                    # For polygons, get points from the polygon() method
                    gds_poly = gdstk.Polygon([(p.x(), p.y()) for p in item.polygon()])
                    total_area += gds_poly.area()
                    total_perimeter += gds_poly.perimeter()
                elif isinstance(item, RectItem):
                    # For rectangles, get the rect() and define its four corner points
                    r = item.rect()
                    pts = [(r.left(), r.top()), (r.right(), r.top()), (r.right(), r.bottom()), (r.left(), r.bottom())]
                    gds_poly = gdstk.Polygon(pts)
                    total_area += gds_poly.area()
                    total_perimeter += gds_poly.perimeter()
                elif isinstance(item, CircleItem):
                    r = item.rect()
                    gds_ellipse = gdstk.ellipse((r.center().x(), r.center().y()), (r.width() / 2, r.height() / 2))
                    total_area += gds_ellipse.area()
                    total_perimeter += gds_ellipse.perimeter()
            except Exception as e:
                print(f"Warning: Measurement failed for an item. {e}")
        
        QMessageBox.information(self, "Measurement",
            f"Selected items: {len(self.scene.selectedItems())}\n"
            f"Total Area: {total_area:.2f} sq. units\n"
            f"Total Perimeter: {total_perimeter:.2f} units")
    
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
        return super().default(o)

if __name__ == "__main__":
    main()
