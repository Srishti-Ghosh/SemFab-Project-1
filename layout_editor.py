import sys
import math
import json
import copy
import uuid
from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Dict, List, Tuple, Optional
from uuid import UUID

# This is a stub for the gdstk library if it's not installed.
# For full functionality (GDS/OAS import/export, fillet), install it: pip install gdstk
try:
    import gdstk
except ImportError:
    print("Warning: gdstk library not found. GDS/OAS and fillet features will be disabled.")
    gdstk = None

# Added for the new 3D view logic
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
    QGraphicsItem, QToolBar, QComboBox, QPushButton, QLabel, QLineEdit,
    QSpinBox, QCheckBox, QDialog, QVBoxLayout, QDialogButtonBox, QToolTip,
    QDockWidget, QListWidget, QListWidgetItem, QWidget, QHBoxLayout, QMessageBox, QDoubleSpinBox, QGraphicsEllipseItem,
    QInputDialog, QTabWidget, QFrame, QToolButton, QGridLayout, QSlider
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
    rect: Tuple[float, float, float, float] # x, y, w, h
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
    layer_by_name: Dict[str, Layer] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.refresh_layer_map()

    def refresh_layer_map(self):
        self.layer_by_name = {l.name: l for l in self.layers}

# -------------------------- Dialogs --------------------------

class GridDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Grid Setup")
        layout = QVBoxLayout()
        self.spacing_input = QLineEdit("50")
        self.size_input_w = QLineEdit("2000")
        self.size_input_h = QLineEdit("1500")
        layout.addWidget(QLabel("Grid Spacing:"))
        layout.addWidget(self.spacing_input)
        layout.addWidget(QLabel("Grid Width:"))
        layout.addWidget(self.size_input_w)
        layout.addWidget(QLabel("Grid Height:"))
        layout.addWidget(self.size_input_h)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_values(self):
        try:
            return int(self.spacing_input.text()), int(self.size_input_w.text()), int(self.size_input_h.text())
        except ValueError:
            return 50, 2000, 1500

class ShapeEditDialog(QDialog):
    def __init__(self, kind, params, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit {kind.capitalize()}")
        self.kind = kind
        self.params = params
        layout = QVBoxLayout()
        self.input_fields = {}

        if kind == "rect":
            for label in ["x", "y", "w", "h"]:
                fld = QLineEdit(f"{params[label]:.2f}")
                layout.addWidget(QLabel(f"{label.upper()}:"))
                layout.addWidget(fld)
                self.input_fields[label] = fld
        elif kind == "circle":
            for label in ["center_x", "center_y", "w", "h"]:
                fld = QLineEdit(f"{params[label]:.2f}")
                layout.addWidget(QLabel(f"{label.replace('_', ' ').title()}:"))
                layout.addWidget(fld)
                self.input_fields[label] = fld
        elif kind == "poly":
            self.pt_edits = []
            for i, (x, y) in enumerate(params["points"]):
                fldx, fldy = QLineEdit(f"{x:.2f}"), QLineEdit(f"{y:.2f}")
                layout.addWidget(QLabel(f"Vertex {i+1} X, Y:"))
                layout.addWidget(fldx); layout.addWidget(fldy)
                self.pt_edits.append((fldx, fldy))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_values(self):
        try:
            if self.kind == "rect": return {k: float(v.text()) for k, v in self.input_fields.items()}
            elif self.kind == "circle": return {k: float(v.text()) for k, v in self.input_fields.items()}
            elif self.kind == "poly": return {"points": [(float(fx.text()), float(fy.text())) for fx, fy in self.pt_edits]}
        except ValueError: return None

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
    """ A custom QGraphicsView for rotating, panning, and zooming a 3D scene. """
    def __init__(self, scene, parent_dialog):
        super().__init__(scene)
        self.parent_dialog = parent_dialog
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self._last_pan_point = QPointF()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
            self._last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            delta = event.pos() - self._last_pan_point
            self.parent_dialog.angle_y += delta.x() * 0.5
            self.parent_dialog.angle_x += delta.y() * 0.5
            self.parent_dialog.draw_3d_view()
        elif event.buttons() & Qt.RightButton:
            # Panning is not implemented in this version to keep rotation simple
            pass
        self._last_pan_point = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoom_in_factor = 1.1
        zoom_out_factor = 1 / zoom_in_factor
        
        if event.angleDelta().y() > 0:
            self.parent_dialog.zoom_scale *= zoom_in_factor
        else:
            self.parent_dialog.zoom_scale *= zoom_out_factor
        
        self.parent_dialog.draw_3d_view()

class ThreeDViewDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive 3D Preview")
        self.setGeometry(100, 100, 800, 800)
        self.project = project

        # 3D View parameters
        self.angle_x = 35.0
        self.angle_y = -45.0
        self.zoom_scale = 1.0
        
        self.layer_thicknesses = {layer.name: 10 for layer in self.project.layers}
        self.sliders = {}

        main_layout = QVBoxLayout(self)

        # --- Controls on Top ---
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_layout = QGridLayout(controls_frame)
        controls_layout.addWidget(QLabel("<b>Layer Thickness</b>"), 0, 0, 1, 3)

        row = 1
        for layer in self.project.layers:
            label = QLabel(layer.name)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(1, 100)
            slider.setValue(self.layer_thicknesses.get(layer.name, 10))
            slider.valueChanged.connect(self.on_slider_change)
            
            value_label = QLabel(f"{slider.value()}")
            slider.value_label = value_label # Attach label to slider

            controls_layout.addWidget(label, row, 0)
            controls_layout.addWidget(slider, row, 1)
            controls_layout.addWidget(value_label, row, 2)
            self.sliders[layer.name] = slider
            row += 1
        
        main_layout.addWidget(controls_frame)

        # --- 3D View at the Bottom ---
        self.scene = QGraphicsScene()
        self.view = Interactive3DView(self.scene, self)
        main_layout.addWidget(self.view)

        self.draw_3d_view()

    def on_slider_change(self):
        for name, slider in self.sliders.items():
            self.layer_thicknesses[name] = slider.value()
            slider.value_label.setText(f"{slider.value()}")
        self.draw_3d_view()

    def project_point(self, x, y, z):
        rad_y = math.radians(self.angle_y)
        rad_x = math.radians(self.angle_x)

        # Rotate around Y-axis
        x1 = x * math.cos(rad_y) + z * math.sin(rad_y)
        z1 = -x * math.sin(rad_y) + z * math.cos(rad_y)
        
        # Rotate around X-axis
        y1 = y * math.cos(rad_x) - z1 * math.sin(rad_x)
        
        return QPointF(x1 * self.zoom_scale, -y1 * self.zoom_scale)

    def draw_3d_view(self):
        if not Path:
            self.scene.clear()
            self.scene.addText("Matplotlib not found. 3D view is disabled.")
            return

        self.scene.clear()

        # 1. Get all polygons from the model, flattened and transformed
        def get_all_polygons(cell, origin=(0, 0), rotation=0, mag=1.0):
            polys = []
            for p_data in cell.polygons:
                if (layer := self.project.layer_by_name.get(p_data.layer)) and layer.visible:
                    rad = math.radians(rotation)
                    cos_r, sin_r = math.cos(rad), math.sin(rad)
                    transformed_points = [
                        (
                            (pt[0] * mag * cos_r - pt[1] * mag * sin_r) + origin[0],
                            (pt[0] * mag * sin_r + pt[1] * mag * cos_r) + origin[1]
                        ) for pt in p_data.points
                    ]
                    polys.append((p_data.layer, transformed_points))
            for ref in cell.references:
                if ref.cell in self.project.cells:
                    # NOTE: This recursive transformation is simplified and works best for non-rotated references.
                    # A full implementation would require matrix composition.
                    new_origin = (origin[0] + ref.origin[0], origin[1] + ref.origin[1])
                    polys.extend(get_all_polygons(self.project.cells[ref.cell], new_origin, ref.rotation + rotation, ref.magnification * mag))
            return polys

        all_polys_by_layer = {layer.name: [] for layer in self.project.layers}
        all_polys = get_all_polygons(self.project.cells[self.project.top])
        for layer_name, points in all_polys:
            if layer_name in all_polys_by_layer:
                all_polys_by_layer[layer_name].append(points)
        
        all_points_flat = [pt for points_list in all_polys_by_layer.values() for points in points_list for pt in points]
        if not all_points_flat: return

        # 2. Define grid and initialize height map
        min_x = min(p[0] for p in all_points_flat)
        max_x = max(p[0] for p in all_points_flat)
        min_y = min(p[1] for p in all_points_flat)
        max_y = max(p[1] for p in all_points_flat)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        grid_step = 25
        
        height_map = {} # Using dict for sparse map {(gx, gy): z}
        all_faces = []

        # 3. Process layers sequentially, building up the height map and generating voxels
        for layer in self.project.layers:
            if not layer.visible: continue
            
            layer_name = layer.name
            thickness = self.layer_thicknesses.get(layer_name, 10)
            color = QColor(*layer.color)
            
            paths = [Path(p) for p in all_polys_by_layer[layer_name]]
            if not paths: continue

            processed_grid_cells = set()

            for gx in range(int(min_x // grid_step), int(max_x // grid_step) + 1):
                for gy in range(int(min_y // grid_step), int(max_y // grid_step) + 1):
                    x, y = gx * grid_step + grid_step / 2, gy * grid_step + grid_step / 2
                    
                    if any(path.contains_point((x, y)) for path in paths):
                        z_base = height_map.get((gx, gy), 0)
                        z_top = z_base + thickness
                        height_map[(gx, gy)] = z_top

                        cx, cy = gx * grid_step - center_x, gy * grid_step - center_y
                        cs = grid_step
                        v = [
                            (cx, cy, z_base), (cx + cs, cy, z_base), (cx + cs, cy + cs, z_base), (cx, cy + cs, z_base),
                            (cx, cy, z_top), (cx + cs, cy, z_top), (cx + cs, cy + cs, z_top), (cx, cy + cs, z_top)
                        ]
                        pv = [self.project_point(*p) for p in v]

                        faces_indices = [[4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
                        
                        for indices in faces_indices:
                            face_2d = [pv[i] for i in indices]
                            face_3d = [v[i] for i in indices]
                            all_faces.append((face_2d, face_3d, color.darker(110) if indices==[0,3,7,4] else color))

        # 4. Sort all collected faces from all layers by depth
        def sort_key(face_data):
            _, points_3d, _ = face_data
            if not points_3d: return -float('inf')
            avg_z = sum(p[2] for p in points_3d) / len(points_3d)
            avg_x = sum(p[0] for p in points_3d) / len(points_3d)
            avg_y = sum(p[1] for p in points_3d) / len(points_3d)
            # Project the centroid of the face and use its Y coordinate as the primary sort key
            return self.project_point(avg_x, avg_y, avg_z).y()

        all_faces.sort(key=sort_key)

        # 5. Draw the sorted faces
        for pts_2d, _, col in all_faces:
            if len(pts_2d) > 2:
                self.scene.addPolygon(QPolygonF(pts_2d), QPen(col.darker(150)), QBrush(col))

        # --- Draw Grid and Axes ---
        model_width = max_x - min_x
        model_depth = max_y - min_y
        model_height = max(height_map.values()) if height_map else max(model_width, model_depth)
        axis_length = max(model_width, model_depth, model_height) * 0.75
        
        grid_pen = QPen(QColor(128, 128, 128, 100))
        for x in range(int(min_x - grid_step), int(max_x + grid_step) + 1, int(grid_step)):
             p1 = self.project_point(x - center_x, min_y - center_y, 0)
             p2 = self.project_point(x - center_x, max_y - center_y, 0)
             self.scene.addLine(QLineF(p1, p2), grid_pen)
        for y in range(int(min_y - grid_step), int(max_y + grid_step) + 1, int(grid_step)):
             p1 = self.project_point(min_x - center_x, y - center_y, 0)
             p2 = self.project_point(max_x - center_x, y - center_y, 0)
             self.scene.addLine(QLineF(p1, p2), grid_pen)

        origin_proj = self.project_point(0, 0, 0)
        
        x_axis_end = self.project_point(axis_length, 0, 0)
        self.scene.addLine(QLineF(origin_proj, x_axis_end), QPen(QColor("red"), 2))
        x_label = self.scene.addText("X"); x_label.setDefaultTextColor(QColor("red")); x_label.setPos(x_axis_end); x_label.setFlag(QGraphicsItem.ItemIgnoresTransformations)

        y_axis_end = self.project_point(0, axis_length, 0)
        self.scene.addLine(QLineF(origin_proj, y_axis_end), QPen(QColor("green"), 2))
        y_label = self.scene.addText("Y"); y_label.setDefaultTextColor(QColor("green")); y_label.setPos(y_axis_end); y_label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        
        z_axis_end = self.project_point(0, 0, axis_length)
        self.scene.addLine(QLineF(origin_proj, z_axis_end), QPen(QColor("blue"), 2))
        z_label = self.scene.addText("Z"); z_label.setDefaultTextColor(QColor("blue")); z_label.setPos(z_axis_end); z_label.setFlag(QGraphicsItem.ItemIgnoresTransformations)

        self.view.setSceneRect(self.scene.itemsBoundingRect().adjusted(-50, -50, 50, 50))

# -------------------------- Graphics items --------------------------

class SceneItemMixin:
    def __init__(self, *args, **kwargs):
        self._drag_start_pos = QPointF()

    def mousePressEvent(self, event):
        self._drag_start_pos = self.scenePos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsItem.ItemIsMovable:
            if self.scenePos() != self._drag_start_pos:
                self.scene().views()[0].parent_win.update_data_from_item_move(self)

class RectItem(QGraphicsRectItem, SceneItemMixin):
    def __init__(self, rect, layer, data_obj, selectable=True):
        QGraphicsRectItem.__init__(self, rect)
        SceneItemMixin.__init__(self)
        self.layer = layer; self.base_color = QColor(*self.layer.color); self.data_obj = data_obj
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable if selectable else 0)
        self.refresh_appearance(selected=False)
        if self.data_obj.name:
            self.setToolTip(self.data_obj.name)

    def mousePressEvent(self, event):
        SceneItemMixin.mousePressEvent(self, event)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        SceneItemMixin.mouseReleaseEvent(self, event)

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
                new_value = snapped_scene_top_left - original_top_left
                return new_value
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        r = self.rect(); dlg = ShapeEditDialog("rect", {"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height()}, self.scene().views()[0])
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
        if self.data_obj.name:
            self.setToolTip(self.data_obj.name)

    def mousePressEvent(self, event):
        SceneItemMixin.mousePressEvent(self, event)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        SceneItemMixin.mouseReleaseEvent(self, event)

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
                new_value = snapped_scene_top_left - original_top_left
                return new_value
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        pts = [(p.x(), p.y()) for p in self.polygon()]
        dlg = ShapeEditDialog("poly", {"points": pts}, self.scene().views()[0])
        if dlg.exec_() == QDialog.Accepted and (vals := dlg.get_values()):
            self.setPolygon(QPolygonF([QPointF(x, y) for x, y in vals["points"]]))
            self.scene().views()[0].parent_win.update_data_from_item_edit(self)

class CircleItem(QGraphicsEllipseItem, SceneItemMixin):
    def __init__(self, rect, layer, data_obj, selectable=True):
        QGraphicsEllipseItem.__init__(self, rect)
        SceneItemMixin.__init__(self)
        self.layer = layer; self.base_color = QColor(*self.layer.color); self.data_obj = data_obj
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable if selectable else 0)
        self.refresh_appearance(selected=False)
        if self.data_obj.name:
            self.setToolTip(self.data_obj.name)

    def mousePressEvent(self, event):
        SceneItemMixin.mousePressEvent(self, event)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        SceneItemMixin.mouseReleaseEvent(self, event)

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
                new_value = snapped_scene_top_left - original_top_left
                return new_value
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        r = self.rect()
        params = {"center_x": r.center().x(), "center_y": r.center().y(), "w": r.width(), "h": r.height()}
        dlg = ShapeEditDialog("circle", params, self.scene().views()[0])
        if dlg.exec_() == QDialog.Accepted and (vals := dlg.get_values()):
            cx, cy, w, h = vals["center_x"], vals["center_y"], vals["w"], vals["h"]
            self.setRect(QRectF(cx - w/2, cy - h/2, w, h))
            self.scene().views()[0].parent_win.update_data_from_item_edit(self)

class RefItem(QGraphicsItem):
    def __init__(self, ref, cell, project, selectable=True):
        super().__init__()
        self.ref, self.cell, self.project = ref, cell, project
        if selectable:
            self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.setPos(QPointF(*self.ref.origin))
        self.setRotation(self.ref.rotation); self.setScale(self.ref.magnification)
        self._draw_children()

    def _draw_children(self):
        for item in self.childItems(): item.setParentItem(None) # Fast clear

        for p_data in self.cell.polygons:
            if (layer := self.project.layer_by_name.get(p_data.layer)) and layer.visible:
                PolyItem(QPolygonF([QPointF(*pt) for pt in p_data.points]), layer, p_data, False).setParentItem(self)
        for e_data in self.cell.ellipses:
            if (layer := self.project.layer_by_name.get(e_data.layer)) and layer.visible:
                CircleItem(QRectF(*e_data.rect), layer, e_data, False).setParentItem(self)
        for r_data in self.cell.references:
            if r_data.cell in self.project.cells:
                RefItem(r_data, self.project.cells[r_data.cell], self.project, selectable=False).setParentItem(self)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            view = self.scene().views()[0]
            if view.parent_win.chk_snap.isChecked():
                return view.snap_point(value)

        if change == QGraphicsItem.ItemPositionHasChanged:
            self.ref.origin = (self.pos().x(), self.pos().y())

        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
             self.scene().views()[0].parent_win._save_state()

    def boundingRect(self): return self.childrenBoundingRect() or QRectF(-5, -5, 10, 10)
    def paint(self, painter, option, widget):
        if self.isSelected():
            painter.setPen(QPen(Qt.darkCyan, 0, Qt.DashLine))
            painter.setBrush(Qt.NoBrush); painter.drawRect(self.boundingRect())

# -------------------------- View/Scene with Rulers --------------------------
class Ruler(QWidget):
    def __init__(self, orientation, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.orientation = orientation
        self.ruler_font = QFont("Arial", 8)
        self.set_size()

    def set_size(self):
        if self.orientation == Qt.Horizontal:
            self.setFixedHeight(30)
        else:
            self.setFixedWidth(30)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setFont(self.ruler_font)

        bg_color = self.palette().color(QPalette.Window)
        tick_color = self.palette().color(QPalette.WindowText)
        text_color = self.palette().color(QPalette.WindowText)

        painter.fillRect(self.rect(), bg_color)

        visible_rect = self.canvas.mapToScene(self.canvas.viewport().rect()).boundingRect()
        zoom = self.canvas.transform().m11()

        base_pitch = self.canvas.parent_win.grid_pitch
        if base_pitch <= 0: return

        minor_pitch = float(base_pitch)
        required_screen_pitch = 8
        while (minor_pitch * zoom) < required_screen_pitch:
            minor_pitch *= 5
        while (minor_pitch * zoom) > required_screen_pitch * 5:
            minor_pitch /= 5
        major_pitch = minor_pitch * 5

        painter.setPen(QPen(tick_color, 1))

        if self.orientation == Qt.Horizontal:
            start = math.floor(visible_rect.left() / minor_pitch) * minor_pitch
            end = math.ceil(visible_rect.right() / minor_pitch) * minor_pitch
            count = int(round(start / minor_pitch))

            x = start
            while x <= end:
                view_x = self.canvas.mapFromScene(QPointF(x, 0)).x()
                tick_height = 10 if count % 5 == 0 else 5
                painter.drawLine(view_x, self.height(), view_x, self.height() - tick_height)

                if count % 5 == 0 and zoom > 0.05:
                       painter.setPen(text_color)
                       painter.drawText(view_x + 2, self.height() - 12, f"{x:.0f}")
                       painter.setPen(tick_color)

                x += minor_pitch
                count += 1
        else: # Vertical
            start_raw = math.floor(visible_rect.bottom() / minor_pitch) * minor_pitch
            end_raw = math.ceil(visible_rect.top() / minor_pitch) * minor_pitch
            start = min(start_raw, end_raw)
            end = max(start_raw, end_raw)

            count = int(round(start / minor_pitch))

            y = start
            while y <= end:
                view_y = self.canvas.mapFromScene(QPointF(0, y)).y()
                tick_width = 10 if count % 5 == 0 else 5
                painter.drawLine(self.width(), view_y, self.width() - tick_width, view_y)

                if count % 5 == 0 and zoom > 0.05:
                    painter.setPen(text_color)
                    painter.save()
                    painter.translate(self.width() - 12, view_y - 2)
                    painter.rotate(-90)
                    painter.drawText(0, 0, f"{y:.0f}")
                    painter.restore()
                    painter.setPen(tick_color)

                y += minor_pitch
                count += 1

class CanvasContainer(QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas

        self.h_ruler = Ruler(Qt.Horizontal, self.canvas)
        self.v_ruler = Ruler(Qt.Vertical, self.canvas)

        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)

        corner = QWidget()
        corner.setFixedSize(30, 30)
        corner.setStyleSheet("background-color: palette(window);")

        layout.addWidget(self.v_ruler, 0, 0)
        layout.addWidget(self.canvas, 0, 1)
        layout.addWidget(corner, 1, 0)
        layout.addWidget(self.h_ruler, 1, 1)

        self.setLayout(layout)

        self.canvas.horizontalScrollBar().valueChanged.connect(self.h_ruler.update)
        self.canvas.verticalScrollBar().valueChanged.connect(self.v_ruler.update)
        self.canvas.zoomChanged.connect(self.h_ruler.update)
        self.canvas.zoomChanged.connect(self.v_ruler.update)

class Canvas(QGraphicsView):
    MODES = {"select": 1, "rect": 2, "poly": 3, "circle": 4, "move": 5}
    zoomChanged = pyqtSignal()

    def __init__(self, scene, get_active_layer, parent=None):
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing); self.setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.mode = self.MODES["select"]; self.start_pos = None; self.temp_item = None
        self.temp_poly_points: List[QPointF] = []
        self.get_active_layer = get_active_layer; self.parent_win = parent
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._panning = False

    def drawBackground(self, painter: QPainter, rect: QRectF):
        super().drawBackground(painter, rect)

        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()

        transform = self.transform()
        zoom = transform.m11()

        bg_color = self.palette().color(self.backgroundRole())
        minor_color = self.palette().color(self.foregroundRole())
        minor_color.setAlphaF(0.2)
        major_color = self.palette().color(self.foregroundRole())
        major_color.setAlphaF(0.4)

        painter.fillRect(rect, bg_color)

        base_pitch = self.parent_win.grid_pitch
        if base_pitch <= 0: return

        minor_pitch = float(base_pitch)
        required_screen_pitch = 8
        while (minor_pitch * zoom) < required_screen_pitch:
            minor_pitch *= 5
        while (minor_pitch * zoom) > required_screen_pitch * 5:
            minor_pitch /= 5
        major_pitch = minor_pitch * 5

        minor_pen = QPen(minor_color, 0)
        major_pen = QPen(major_color, 0)

        left, right = visible_rect.left(), visible_rect.right()
        top, bottom = visible_rect.top(), visible_rect.bottom()

        x_start = max(0, math.floor(left / minor_pitch) * minor_pitch)
        x_end = max(0, math.ceil(right / minor_pitch) * minor_pitch)
        count = int(round(x_start / minor_pitch))
        x = x_start
        while x <= x_end:
            painter.setPen(major_pen if count % 5 == 0 else minor_pen)
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            x += minor_pitch
            count += 1

        y_start_raw = math.floor(bottom / minor_pitch) * minor_pitch
        y_end_raw = math.ceil(top / minor_pitch) * minor_pitch
        y_start = min(y_start_raw, y_end_raw)
        y_end = max(y_start_raw, y_end_raw)

        count = int(round(y_start / minor_pitch))
        y = y_start
        while y <= y_end:
            if y < 0:
                y += minor_pitch
                count += 1
                continue
            painter.setPen(major_pen if count % 5 == 0 else minor_pen)
            painter.drawLine(QPointF(left, y), QPointF(right, y))
            y += minor_pitch
            count += 1

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        old_pos = self.mapToScene(event.pos())

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        self.zoomChanged.emit()

    def set_mode(self, mode_name: str):
        self.mode = self.MODES.get(mode_name, self.MODES["select"])
        is_move_mode = (self.mode == self.MODES["move"])
        is_select_mode = (self.mode == self.MODES["select"])

        if is_select_mode:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        else:
            self.setDragMode(QGraphicsView.NoDrag)

        for item in self.scene().items():
            if item.flags() & QGraphicsItem.ItemIsSelectable:
                item.setFlag(QGraphicsItem.ItemIsMovable, is_move_mode or is_select_mode)

        self.temp_cancel()

    def temp_cancel(self):
        if self.temp_item: self.scene().removeItem(self.temp_item)
        self.temp_item = None; self.temp_poly_points.clear()
        self.viewport().setCursor(Qt.ArrowCursor)

    def snap_point(self, p: QPointF) -> QPointF:
        if not self.parent_win.chk_snap.isChecked(): return p
        pitch = self.parent_win.grid_pitch
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
        self.parent_win.statusBar().showMessage(f"X: {scene_pos.x():.2f}, Y: {scene_pos.y():.2f}")
        if self._panning:
            delta = self._pan_start - event.pos(); self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
            event.accept(); return

        if self.temp_item and self.start_pos:
            self.temp_item.setRect(QRectF(self.start_pos, self.snap_point(scene_pos)).normalized())
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
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.mode == self.MODES["poly"] and event.button() == Qt.LeftButton:
            self.finish_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape: self.parent_win.set_active_tool("select"); self.temp_cancel()
        elif event.key() == Qt.Key_Delete: self.parent_win.delete_selected_items()
        else: super().keyPressEvent(event)

    def finish_polygon(self):
        can_create_poly = len(self.temp_poly_points) >= 3 and (layer := self.get_active_layer())

        if can_create_poly:
            final_poly = QPolygonF(self.temp_poly_points)

        self.temp_cancel()

        if can_create_poly:
            self.parent_win.add_poly_to_active_cell(final_poly, layer)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handles dropping a cell from the list onto the canvas."""
        cell_name = event.mimeData().text()
        if not (cell_name and cell_name in self.parent_win.project.cells):
            event.ignore()
            return
        
        if cell_name == self.parent_win.active_cell_name:
            QToolTip.showText(event.globalPos(), "Cannot drop a cell onto itself.")
            event.ignore()
            return

        item_at_pos = self.itemAt(event.pos())
        target_data_obj = None

        if hasattr(item_at_pos, 'data_obj') and (item_at_pos.flags() & QGraphicsItem.ItemIsSelectable):
            if isinstance(item_at_pos, RectItem) or \
               (isinstance(item_at_pos, PolyItem) and self.parent_win._is_axis_aligned_rect(item_at_pos.data_obj.points)):
                target_data_obj = item_at_pos.data_obj

        if target_data_obj:
            self.parent_win.instantiate_cell_on_shape(cell_name, target_data_obj)
            event.acceptProposedAction()
        else:
            scene_pos = self.mapToScene(event.pos())
            self.parent_win.add_reference_to_active_cell(cell_name, self.snap_point(scene_pos))
            event.acceptProposedAction()

# -------------------------- Main window --------------------------

class MainWindow(QMainWindow):
    def __init__(self, **kwargs):
        super().__init__()
        self.setWindowTitle("2D Mask Layout Editor")
        self.grid_pitch, self.grid_width, self.grid_height = kwargs.get("grid_pitch", 50), kwargs.get("canvas_width", 2000), kwargs.get("canvas_height", 1500)

        self.project = self._create_default_project()
        self.active_cell_name, self.active_layer_name = self.project.top, self.project.layers[0].name if self.project.layers else None

        self.scene = QGraphicsScene(0, 0, self.grid_width, self.grid_height)
        self.view = Canvas(self.scene, self.active_layer, self)
        self.view.scale(1, -1)
        self.view.translate(0, -self.grid_height)

        self.canvas_container = CanvasContainer(self.view, self)
        self.setCentralWidget(self.canvas_container)

        self._undo_stack, self._redo_stack = [], []
        self._build_ui()
        self._apply_stylesheet()
        self.statusBar().showMessage("Ready")
        self.update_ui_from_project(); self._save_state()

    def _create_themed_icon(self, icon_name):
        """
        Creates a QIcon from an SVG file that adapts to light/dark themes.
        Requires an 'icons' folder in the same directory as the script.
        """
        try:
            pixmap = QPixmap(f"icons/{icon_name}.svg")
            is_dark = self.palette().color(QPalette.Window).value() < 128

            if is_dark:
                painter = QPainter(pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                painter.fillRect(pixmap.rect(), Qt.white)
                painter.end()
            return QIcon(pixmap)
        except Exception:
            # Fallback icon if SVG is not found
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.gray, 2))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, icon_name[0].upper())
            painter.end()
            return QIcon(pixmap)


    def _create_default_project(self):
        return Project(layers=[Layer("N-doping", (0, 0, 255)), Layer("P-doping", (255, 0, 0)), Layer("Oxide", (0, 200, 0)), Layer("Metal", (200, 200, 200)), Layer("Contact", (255, 200, 0))],
                       cells={"TOP": Cell()}, top="TOP")

    def _build_ui(self):
        self.tool_buttons = {}
        self._build_menus()
        self._build_tool_tabs()
        self._build_cell_dock()
        self._build_layer_dock()
        self.set_active_tool("select")

    def _build_menus(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction("New", self.new_doc, "Ctrl+N")
        file_menu.addAction("Open JSON…", self.open_json, "Ctrl+O")
        file_menu.addAction("Save JSON…", self.save_json, "Ctrl+S")
        file_menu.addSeparator()
        if gdstk:
            file_menu.addAction("Import GDS/OAS…", self.open_gds_oas)
            file_menu.addAction("Export GDS…", self.save_gds)
            file_menu.addAction("Export OAS…", self.save_oas)
        file_menu.addAction("Export SVG…", self.export_svg)
        file_menu.addSeparator()
        file_menu.addAction("Undo", self.undo, "Ctrl+Z")
        file_menu.addAction("Redo", self.redo, "Ctrl+Y")
        file_menu.addSeparator(); file_menu.addAction("Quit", self.close, "Ctrl+Q")

    def _build_tool_tabs(self):
        dock = QDockWidget(self)
        dock.setTitleBarWidget(QWidget())
        dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetMovable)

        tabs = QTabWidget()
        dock.setWidget(tabs)

        main_tab = QWidget()
        main_layout = QHBoxLayout(main_tab)
        main_layout.setAlignment(Qt.AlignLeft)
        main_layout.setContentsMargins(0, 5, 0, 5)

        main_layout.addWidget(self._create_ribbon_group("Selection", [
            self._create_tool_button("select", "Select", "mouse-pointer"),
            self._create_tool_button("move", "Move", "move")
        ]))
        main_layout.addWidget(self._create_ribbon_group("History", [
            self._create_action_button("Delete", "trash-2", self.delete_selected_items),
            self._create_action_button("Undo", "corner-up-left", self.undo),
            self._create_action_button("Redo", "corner-up-right", self.redo)
        ]))

        grid_widget = QWidget()
        grid_layout = QVBoxLayout(grid_widget)
        self.chk_snap = QCheckBox("Snap"); self.chk_snap.setChecked(True)
        self.spin_grid = QSpinBox(); self.spin_grid.setRange(1, 1000)
        self.spin_grid.valueChanged.connect(self.set_grid_pitch)
        grid_layout.addWidget(self.chk_snap)
        grid_size_layout = QHBoxLayout()
        grid_size_layout.addWidget(QLabel("Grid:"))
        grid_size_layout.addWidget(self.spin_grid)
        grid_layout.addLayout(grid_size_layout)
        main_layout.addWidget(self._create_ribbon_group("Grid", [grid_widget]))

        main_layout.addStretch()
        tabs.addTab(main_tab, "Main")

        draw_tab = QWidget()
        draw_layout = QHBoxLayout(draw_tab)
        draw_layout.setAlignment(Qt.AlignLeft)
        draw_layout.setContentsMargins(0, 5, 0, 5)
        draw_layout.addWidget(self._create_ribbon_group("Create", [
            self._create_tool_button("rect", "Rect", "square"),
            self._create_tool_button("circle", "Circle", "circle"),
            self._create_tool_button("poly", "Poly", "pen-tool")
        ]))
        draw_layout.addWidget(self._create_ribbon_group("Modify", [
            self._create_action_button("Rename", "tag", self.rename_selected_shape),
            self._create_action_button("Re-snap", "grid", self.resnap_all_items_to_grid),
            self._create_action_button("Finish Poly", "check-square", self.view.finish_polygon),
            self._create_action_button("Fillet", "git-commit", self.fillet_selected_poly)
        ]))
        draw_layout.addStretch()
        tabs.addTab(draw_tab, "Draw")

        view_tab = QWidget()
        view_layout = QHBoxLayout(view_tab)
        view_layout.setAlignment(Qt.AlignLeft)
        view_layout.setContentsMargins(0, 5, 0, 5)
        view_layout.addWidget(self._create_ribbon_group("Display", [
            self._create_action_button("Fill", "maximize", self.fill_view),
            self._create_action_button("3D View", "box", self.show_3d_view),
            self._create_action_button("Measure", "compass", self.show_measurements)
        ]))
        view_layout.addStretch()
        tabs.addTab(view_tab, "View")

        self.addDockWidget(Qt.TopDockWidgetArea, dock)

    def _create_tool_button(self, tool_name, text, icon_name):
        button = QToolButton()
        button.setText(text)
        button.setIcon(self._create_themed_icon(icon_name))
        button.setIconSize(QSize(24, 24))
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setCheckable(True)
        button.clicked.connect(lambda: self.set_active_tool(tool_name))
        self.tool_buttons[tool_name] = button
        return button

    def _create_action_button(self, text, icon_name, slot):
        button = QToolButton()
        button.setText(text)
        button.setIcon(self._create_themed_icon(icon_name))
        button.setIconSize(QSize(24, 24))
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.clicked.connect(slot)
        return button

    def _create_ribbon_group(self, title, widgets):
        group_container = QWidget()
        group_layout = QHBoxLayout(group_container)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(0)

        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(10, 5, 10, 5)
        buttons_layout.setSpacing(5)

        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)
        panel_layout.setContentsMargins(0,0,0,0)
        panel_layout.setSpacing(2)
        panel_layout.setAlignment(Qt.AlignBottom)

        for widget in widgets:
            buttons_layout.addWidget(widget)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)

        panel_layout.addWidget(buttons_widget)
        panel_layout.addWidget(title_label)

        group_layout.addWidget(panel_widget)

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        group_layout.addWidget(separator)

        return group_container

    def set_active_tool(self, tool_name):
        self.view.set_mode(tool_name)
        for name, btn in self.tool_buttons.items():
            btn.setChecked(name == tool_name)

    def _build_cell_dock(self):
        dock = QDockWidget("Cells", self)
        self.list_cells = QListWidget()
        self.list_cells.setDragEnabled(True)
        self.list_cells.itemDoubleClicked.connect(self._on_cell_double_clicked)

        widget, layout = QWidget(), QVBoxLayout()
        layout.addWidget(QLabel("Double-click to edit, Drag to instantiate"))
        layout.addWidget(self.list_cells)
        btns = QHBoxLayout()
        for label, func in [("Add", self.add_cell_dialog), ("Rename", self.rename_cell_dialog), ("Delete", self.delete_cell_dialog)]:
            btn = QPushButton(label)
            btn.clicked.connect(func)
            btns.addWidget(btn)
        layout.addLayout(btns)
        widget.setLayout(layout)
        dock.setWidget(widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _build_layer_dock(self):
        dock = QDockWidget("Layers", self)
        self.list_layers = QListWidget()
        self.list_layers.itemChanged.connect(self._update_layer_visibility)
        self.list_layers.itemSelectionChanged.connect(self._on_layer_selected)
        self.list_layers.itemDoubleClicked.connect(self.change_layer_color_dialog)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Toggle visibility, double-click color"))
        layout.addWidget(self.list_layers)

        btns1 = QHBoxLayout()
        layer_actions = [
            ("Move to Top (drawn last)", "chevrons-up", self.move_layer_to_top),
            ("Move Forward", "chevron-up", self.move_layer_forward),
            ("Move Backward", "chevron-down", self.move_layer_backward),
            ("Move to Bottom (drawn first)", "chevrons-down", self.move_layer_to_bottom)
        ]

        for tooltip, icon_name, func in layer_actions:
            btn = QToolButton()
            btn.setIcon(self._create_themed_icon(icon_name))
            btn.setToolTip(tooltip)
            btn.clicked.connect(func)
            btns1.addWidget(btn)

        layout.addLayout(btns1)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        btns2 = QHBoxLayout()
        for label, func in [("Add", self.add_layer_dialog), ("Rename", self.rename_layer_dialog), ("Delete", self.delete_layer_dialog)]:
            btn = QPushButton(label)
            btn.clicked.connect(func)
            btns2.addWidget(btn)
        layout.addLayout(btns2)

        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def update_ui_from_project(self):
        if self.project.top not in self.project.cells: self.project.top = list(self.project.cells.keys())[0] if self.project.cells else None
        if not self.active_cell_name or self.active_cell_name not in self.project.cells:
             self.active_cell_name = self.project.top
        self.spin_grid.setValue(self.grid_pitch)
        self._refresh_cell_list(); self._refresh_layer_list(); self._redraw_scene()

    def _refresh_cell_list(self):
        self.list_cells.blockSignals(True)
        current_text = self.active_cell_name
        self.list_cells.clear()
        for name in sorted(self.project.cells.keys()):
            item = QListWidgetItem(name); self.list_cells.addItem(item)
            if name == current_text: item.setSelected(True)
        self.list_cells.blockSignals(False)

    def _refresh_layer_list(self):
        self.list_layers.blockSignals(True)
        current_text = self.active_layer_name
        self.list_layers.clear()
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
        if item.text() != self.active_cell_name:
            self.active_cell_name = item.text()
            self._redraw_scene()

    def _on_layer_selected(self):
        if item := self.list_layers.currentItem(): self.active_layer_name = item.text()

    def _update_layer_visibility(self, item):
        if (layer := self.project.layer_by_name.get(item.text())):
            layer.visible = item.checkState() == Qt.Checked
            self._redraw_scene(); self._save_state()

    def active_layer(self): return self.project.layer_by_name.get(self.active_layer_name)

    def _save_state(self):
        self._redo_stack.clear()
        self._undo_stack.append(copy.deepcopy(self.project))
        if len(self._undo_stack) > 50: self._undo_stack.pop(0)

    def undo(self):
        if len(self._undo_stack) > 1:
            self._redo_stack.append(self._undo_stack.pop())
            self.project = copy.deepcopy(self._undo_stack[-1])
            self.update_ui_from_project()

    def redo(self):
        if self._redo_stack:
            self.project = self._redo_stack.pop()
            self._undo_stack.append(copy.deepcopy(self.project))
            self.update_ui_from_project()

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
            xs = [p[0] for p in target_shape_data.points]
            ys = [p[1] for p in target_shape_data.points]
            placeholder_rect = QRectF(QPointF(min(xs), min(ys)), QPointF(max(xs), max(ys)))
        elif isinstance(target_shape_data, Ellipse):
            placeholder_rect = QRectF(*target_shape_data.rect)
        else:
            return

        dropped_cell_bounds = self.get_cell_bounds(cell_name_to_drop)
        magnification = 1.0
        
        if not dropped_cell_bounds.isNull() and dropped_cell_bounds.width() > 0 and dropped_cell_bounds.height() > 0:
            scale_x = placeholder_rect.width() / dropped_cell_bounds.width()
            scale_y = placeholder_rect.height() / dropped_cell_bounds.height()
            magnification = min(scale_x, scale_y)

        origin = placeholder_rect.center() - (dropped_cell_bounds.center() * magnification)

        active_cell = self.project.cells[self.active_cell_name]
        
        # FIX: Remove placeholder by its unique ID, not by reference
        if isinstance(target_shape_data, Poly):
            active_cell.polygons = [p for p in active_cell.polygons if p.uuid != target_shape_data.uuid]
        elif isinstance(target_shape_data, Ellipse):
            active_cell.ellipses = [e for e in active_cell.ellipses if e.uuid != target_shape_data.uuid]
        
        new_ref = Ref(cell=cell_name_to_drop, origin=(origin.x(), origin.y()), magnification=magnification)
        active_cell.references.append(new_ref)

        self._save_state()
        self._redraw_scene()

    def update_data_from_item_move(self, item):
        drag_start_pos = item._drag_start_pos
        final_snapped_pos = self.view.snap_point(item.pos())
        dx = final_snapped_pos.x() - drag_start_pos.x()
        dy = final_snapped_pos.y() - drag_start_pos.y()

        if abs(dx) < 1e-9 and abs(dy) < 1e-9: return

        if isinstance(item, RefItem):
            old_origin = item.ref.origin
            item.ref.origin = (old_origin[0] + dx, old_origin[1] + dy)
        elif isinstance(item, CircleItem):
            old_rect = item.data_obj.rect
            item.data_obj.rect = (old_rect[0] + dx, old_rect[1] + dy, old_rect[2], old_rect[3])
        elif isinstance(item, (RectItem, PolyItem)):
            item.data_obj.points = [(p[0] + dx, p[1] + dy) for p in item.data_obj.points]

        self._save_state()
        self._redraw_scene()

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
        self._redraw_scene()

    def delete_selected_items(self):
        cell = self.project.cells.get(self.active_cell_name)
        if not cell: return

        selected_items = self.scene.selectedItems()
        if not selected_items: return

        data_objects_to_delete = []
        for item in selected_items:
            if hasattr(item, 'data_obj'):
                data_objects_to_delete.append(item.data_obj)
            elif isinstance(item, RefItem):
                data_objects_to_delete.append(item.ref)

        if data_objects_to_delete:
            self.scene.clearSelection()

            cell.polygons = [p for p in cell.polygons if p not in data_objects_to_delete]
            cell.ellipses = [e for e in cell.ellipses if e not in data_objects_to_delete]
            cell.references = [r for r in cell.references if r not in data_objects_to_delete]

            self._redraw_scene()
            self._save_state()

    def fillet_selected_poly(self):
        if not gdstk:
            QMessageBox.warning(self, "Feature Disabled", "Please install the 'gdstk' library to use this feature.")
            return
        selected = [it for it in self.scene.selectedItems() if isinstance(it, PolyItem)]
        if not selected:
            self.statusBar().showMessage("Select one or more polygons to fillet.")
            return

        dlg = FilletDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            radius = dlg.get_radius()
            for item in selected:
                try:
                    gds_poly = gdstk.Polygon([(p.x(), p.y()) for p in item.polygon()])
                    gds_poly.fillet(radius)
                    new_points = gds_poly.points.tolist()
                    item.data_obj.points = new_points
                except Exception as e:
                    QMessageBox.warning(self, "Fillet Error", f"Could not fillet polygon: {e}")

            self._save_state()
            self._redraw_scene()

    def show_measurements(self):
        selected = self.scene.selectedItems()
        if not selected:
            self.statusBar().showMessage("No item selected.")
            return

        msg = ""
        for item in selected:
            if isinstance(item, RectItem):
                r = item.rect()
                msg += f"Rectangle ({item.layer.name}):\nx={r.x():.2f}, y={r.y():.2f}, w={r.width():.2f}, h={r.height():.2f}\n\n"
            elif isinstance(item, PolyItem):
                pts = [(p.x(), p.y()) for p in item.polygon()]
                msg += f"Polygon ({item.layer.name}) with {len(pts)} vertices.\n\n"
            elif isinstance(item, CircleItem):
                r = item.rect()
                center = (r.center().x(), r.center().y())
                msg += f"Circle ({item.layer.name}):\ncenter=({center[0]:.2f}, {center[1]:.2f}), w={r.width():.2f}, h={r.height():.2f}\n\n"
            elif isinstance(item, RefItem):
                pos = item.pos()
                msg += f"Reference to '{item.ref.cell}':\nOrigin=({pos.x():.2f}, {pos.y():.2f}), Rotation={item.ref.rotation:.2f}°, Mag={item.ref.magnification:.3f}\n\n"

        if msg:
            QMessageBox.information(self, "Measurements", msg)

    def fill_view(self):
        if not self.scene.items():
            return
        items_rect = self.scene.itemsBoundingRect()
        if not items_rect.isNull():
            margin_x = items_rect.width() * 0.05
            margin_y = items_rect.height() * 0.05
            self.view.fitInView(items_rect.adjusted(-margin_x, -margin_y, margin_x, margin_y), Qt.KeepAspectRatio)
            self.view.zoomChanged.emit()

    def show_3d_view(self): ThreeDViewDialog(self.project, self).exec_()

    def new_doc(self):
        self.project = self._create_default_project(); self.update_ui_from_project(); self._save_state()

    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON Files (*.json)")
        if path:
            # Custom encoder to handle UUID objects
            class ProjectEncoder(json.JSONEncoder):
                def default(self, o):
                    if isinstance(o, UUID):
                        return str(o)
                    if is_dataclass(o):
                        return asdict(o)
                    return super().default(o)
            with open(path, "w") as f:
                json.dump(self.project, f, indent=2, cls=ProjectEncoder)


    def open_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open JSON", "", "JSON Files (*.json)")
        if path:
            with open(path, "r") as f: data = json.load(f)
            self.project = Project(units=data.get('units', 'um'), top=data.get('top', 'TOP'),
                                   layers=[Layer(**l) for l in data.get('layers', [])])
            for name, c_data in data.get('cells', {}).items():
                polys_data = c_data.get('polygons', [])
                for p in polys_data: p.pop('uuid', None) # Remove old uuid if it exists
                polys = [Poly(**p) for p in polys_data]

                ellipses_data = c_data.get('ellipses', [])
                for e in ellipses_data: e.pop('uuid', None)
                ellipses = [Ellipse(**e) for e in ellipses_data]
                
                self.project.cells[name] = Cell(
                    polygons=polys,
                    ellipses=ellipses,
                    references=[Ref(**r) for r in c_data.get('references', [])]
                )
            self.project.refresh_layer_map()
            self.update_ui_from_project(); self._save_state()

    def export_svg(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", "", "SVG Files (*.svg)")
        if not path: return
        gen = QSvgGenerator(); gen.setFileName(path)
        rect = self.scene.itemsBoundingRect().adjusted(-50, -50, 50, 50)
        gen.setViewBox(rect); painter = QPainter(gen)
        self.scene.render(painter, target=QRectF(), source=rect); painter.end()

    def open_gds_oas(self):
        if not gdstk:
            QMessageBox.warning(self, "Feature Disabled", "Please install the 'gdstk' library to use this feature.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open GDS/OAS", "", "Layout Files (*.gds *.oas)")
        if not path: return
        try:
            lib = gdstk.read_gds(path) if path.lower().endswith('.gds') else gdstk.read_oas(path)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to read file: {e}")
            return

        self.project = Project(units=lib.unit)
        gds_layers = set()
        for cell in lib.cells:
            for poly in cell.polygons: gds_layers.add(poly.layer)

        self.project.layers = [Layer(f"Layer_{l}", (l*20%255, l*50%255, l*80%255)) for l in sorted(list(gds_layers))]
        self.project.refresh_layer_map()

        for cell in lib.cells:
            new_cell = Cell()
            for poly in cell.polygons:
                new_cell.polygons.append(Poly(f"Layer_{poly.layer}", poly.points.tolist()))
            for ref in cell.references:
                ref_cell_name = ref.cell.name if isinstance(ref.cell, gdstk.Cell) else ref.cell
                new_cell.references.append(Ref(ref_cell_name, ref.origin, ref.rotation, ref.magnification))
            self.project.cells[cell.name] = new_cell

        self.project.top = lib.top_level()[0].name if lib.top_level() else (list(self.project.cells.keys())[0] if self.project.cells else None)
        self.update_ui_from_project()
        self._save_state()

    def save_gds(self): self._write_gdstk(False)
    def save_oas(self): self._write_gdstk(True)

    def _write_gdstk(self, is_oas):
        if not gdstk:
            QMessageBox.warning(self, "Feature Disabled", "Please install the 'gdstk' library to use this feature.")
            return
        file_ext = "*.oas" if is_oas else "*.gds"
        path, _ = QFileDialog.getSaveFileName(self, f"Save {file_ext.upper()}", "", f"Files ({file_ext})")
        if not path: return

        try:
            lib = gdstk.Library(unit=float(self.project.units.replace('um','e-6')))
            layer_map = {layer.name: i for i, layer in enumerate(self.project.layers, 1)}

            gds_cells = {name: lib.new_cell(name) for name in self.project.cells}

            for name, cell_data in self.project.cells.items():
                gds_cell = gds_cells[name]
                for p in cell_data.polygons: gds_cell.add(gdstk.Polygon(p.points, layer=layer_map.get(p.layer, 0)))
                for e in cell_data.ellipses:
                    r = e.rect; center = (r[0] + r[2]/2, r[1] + r[3]/2); radius = (r[2]/2)
                    gds_cell.add(gdstk.ellipse(center, radius, layer=layer_map.get(e.layer, 0)))
                for r in cell_data.references:
                    if r.cell in gds_cells:
                        gds_cell.add(gdstk.Reference(gds_cells[r.cell], r.origin, r.rotation, r.magnification))

            if is_oas: lib.write_oas(path)
            else: lib.write_gds(path)
            self.statusBar().showMessage(f"Saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to write file: {e}")

    def set_grid_pitch(self, value):
        self.grid_pitch = value
        self.view.viewport().update()

    def resnap_all_items_to_grid(self):
        reply = QMessageBox.question(self, 'Re-snap Cell',
                                     "This will snap all shapes in the current cell to the new grid pitch. This action can be undone. Continue?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        new_pitch = self.grid_pitch
        if new_pitch <= 0: return

        active_cell = self.project.cells.get(self.active_cell_name)
        if not active_cell: return

        def snap(val):
            return round(val / new_pitch) * new_pitch

        for poly in active_cell.polygons:
            poly.points = [(snap(p[0]), snap(p[1])) for p in poly.points]

        for ellipse in active_cell.ellipses:
            x, y, w, h = ellipse.rect
            new_x = snap(x)
            new_y = snap(y)
            new_w = max(new_pitch, snap(w))
            new_h = max(new_pitch, snap(h))
            ellipse.rect = (new_x, new_y, new_w, new_h)

        for ref in active_cell.references:
            ox, oy = ref.origin
            ref.origin = (snap(ox), snap(oy))

        self._save_state()
        self._redraw_scene()

    def add_cell_dialog(self):
        name, ok = QInputDialog.getText(self, "Add Cell", "Cell name:")
        if ok and name and name not in self.project.cells:
            self.project.cells[name] = Cell(); self._refresh_cell_list(); self._save_state()

    def rename_cell_dialog(self):
        if not (sel := self.list_cells.selectedItems()): return
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
        if not (sel := self.list_cells.selectedItems()): return
        name = sel[0].text()
        if len(self.project.cells) <= 1:
            QMessageBox.warning(self, "Delete Error", "Cannot delete the last cell."); return
        if name == self.project.top:
            QMessageBox.warning(self, "Delete Error", "Cannot delete the top-level cell."); return

        for cell_name, cell in self.project.cells.items():
            if any(ref.cell == name for ref in cell.references):
                QMessageBox.warning(self, "Delete Error", f"Cannot delete cell '{name}' as it is referenced by '{cell_name}'."); return

        del self.project.cells[name]
        if self.active_cell_name == name: self.active_cell_name = self.project.top
        self._refresh_cell_list(); self._redraw_scene(); self._save_state()

    def add_layer_dialog(self):
        name, ok = QInputDialog.getText(self, "Add Layer", "Layer name:")
        if ok and name and name not in self.project.layer_by_name:
            color = QColorDialog.getColor()
            if color.isValid():
                new_layer = Layer(name, (color.red(), color.green(), color.blue()))
                self.project.layers.append(new_layer); self.project.refresh_layer_map()
                self._refresh_layer_list(); self._save_state()

    def rename_layer_dialog(self):
        if not (sel := self.list_layers.currentItem()): return
        old_name = sel.text()
        new_name, ok = QInputDialog.getText(self, "Rename Layer", "New name:", text=old_name)
        if ok and new_name and new_name != old_name and new_name not in self.project.layer_by_name:
            for cell in self.project.cells.values():
                for p in cell.polygons:
                    if p.layer == old_name: p.layer = new_name
                for e in cell.ellipses:
                    if e.layer == old_name: e.layer = new_name

            layer_obj = self.project.layer_by_name[old_name]
            layer_obj.name = new_name
            self.project.refresh_layer_map()
            if self.active_layer_name == old_name: self.active_layer_name = new_name
            self._refresh_layer_list(); self._redraw_scene(); self._save_state()

    def delete_layer_dialog(self):
        if not (sel := self.list_layers.currentItem()): return
        name = sel.text()
        for cell_name, cell in self.project.cells.items():
            if any(p.layer == name for p in cell.polygons) or any(e.layer == name for e in cell.ellipses):
                QMessageBox.warning(self, "Delete Error", f"Layer '{name}' is in use in cell '{cell_name}'."); return

        self.project.layers = [l for l in self.project.layers if l.name != name]
        self.project.refresh_layer_map()
        if self.active_layer_name == name: self.active_layer_name = self.project.layers[0].name if self.project.layers else None
        self._refresh_layer_list(); self._save_state()

    def move_layer_forward(self): self._move_layer(1)
    def move_layer_backward(self): self._move_layer(-1)

    def move_layer_to_top(self):
        if not (sel := self.list_layers.currentItem()): return
        if not (layer := self.project.layer_by_name.get(sel.text())): return

        if self.project.layers.index(layer) < len(self.project.layers) - 1:
            self.project.layers.remove(layer)
            self.project.layers.append(layer)
            self._save_state()
            self._refresh_layer_list()
            self._redraw_scene()

    def move_layer_to_bottom(self):
        if not (sel := self.list_layers.currentItem()): return
        if not (layer := self.project.layer_by_name.get(sel.text())): return

        if self.project.layers.index(layer) > 0:
            self.project.layers.remove(layer)
            self.project.layers.insert(0, layer)
            self._save_state()
            self._refresh_layer_list()
            self._redraw_scene()

    def _move_layer(self, direction):
        if not (sel := self.list_layers.currentItem()): return
        if not (layer := self.project.layer_by_name.get(sel.text())): return

        try:
            idx = self.project.layers.index(layer)
            new_idx = idx + direction
            if 0 <= new_idx < len(self.project.layers):
                self.project.layers.pop(idx)
                self.project.layers.insert(new_idx, layer)
                self._save_state()
                self._refresh_layer_list()
                self._redraw_scene()
        except ValueError:
            return

    def change_layer_color_dialog(self, item):
        layer = self.project.layer_by_name.get(item.text())
        if layer:
            color = QColorDialog.getColor(QColor(*layer.color))
            if color.isValid():
                layer.color = (color.red(), color.green(), color.blue())
                self._refresh_layer_list(); self._redraw_scene(); self._save_state()

    def _redraw_scene(self):
        self.scene.clear()
        if self.active_cell_name: self._draw_active_cell()
        # Restore the correct tool state after redrawing
        if hasattr(self.view, 'mode'):
            current_mode_name = next((name for name, val in self.view.MODES.items() if val == self.view.mode), "select")
            self.set_active_tool(current_mode_name)


    def _is_axis_aligned_rect(self, points):
        if len(points) != 4: return False
        xs = set(p[0] for p in points)
        ys = set(p[1] for p in points)
        return len(xs) == 2 and len(ys) == 2

    def _draw_active_cell(self):
        if not (cell := self.project.cells.get(self.active_cell_name)): return

        for layer in self.project.layers:
            if not layer.visible:
                continue

            for p_data in cell.polygons:
                if p_data.layer != layer.name:
                    continue

                if self._is_axis_aligned_rect(p_data.points):
                    xs = [p[0] for p in p_data.points]
                    ys = [p[1] for p in p_data.points]
                    x_min, y_min = min(xs), min(ys)
                    width, height = max(xs) - x_min, max(ys) - y_min
                    self.scene.addItem(RectItem(QRectF(x_min, y_min, width, height), layer, p_data))
                else:
                    self.scene.addItem(PolyItem(QPolygonF([QPointF(*pt) for pt in p_data.points]), layer, p_data))

            for e_data in cell.ellipses:
                if e_data.layer == layer.name:
                    self.scene.addItem(CircleItem(QRectF(*e_data.rect), layer, e_data))

        for r_data in cell.references:
            if r_data.cell in self.project.cells:
                self.scene.addItem(RefItem(r_data, self.project.cells[r_data.cell], self.project))

    def _apply_stylesheet(self):
        stylesheet = """
            QMainWindow, #ContentArea, QDockWidget {
                background-color: palette(window);
                color: palette(window-text);
            }
            #ContentArea { font-size: 24px; border: none; }
            QTabWidget::pane {
                border-top: 1px solid palette(mid);
                background: palette(base);
            }
            QTabBar {
                alignment: left;
            }
            QTabBar::tab {
                background: transparent; border: none;
                padding: 8px 15px; font-size: 9pt;
                color: palette(text); margin-right: 1px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:hover {
                background: palette(light);
            }
            QTabBar::tab:selected {
                background: palette(base);
                color: palette(highlight);
            }
            QToolButton {
                border: 1px solid transparent; padding: 5px; border-radius: 4px;
                color: palette(text);
                background-color: transparent;
            }
            QToolButton:hover {
                background-color: palette(light);
                border: 1px solid palette(mid);
            }
            QToolButton:pressed, QToolButton:checked {
                background-color: palette(midlight);
                border: 1px solid palette(mid);
            }
            QLabel, QCheckBox {
                color: palette(text);
                font-size: 8pt;
            }
            QFrame[frameShape="5"] { /* VLine */
                color: palette(midlight);
            }
            QMenuBar { background-color: palette(window); }
            QMenuBar::item:selected { background: palette(highlight); }
        """
        self.setStyleSheet(stylesheet)

    def rename_selected_shape(self):
        selected_items = self.scene.selectedItems()
        if len(selected_items) != 1:
            self.statusBar().showMessage("Please select a single shape to rename.")
            return

        item = selected_items[0]
        if not hasattr(item, 'data_obj') or not hasattr(item.data_obj, 'name'):
            self.statusBar().showMessage("Selected item cannot be named.")
            return

        current_name = item.data_obj.name or ""
        new_name, ok = QInputDialog.getText(self, "Rename Shape", "Enter name:", text=current_name)

        if ok:
            item.data_obj.name = new_name if new_name else None
            item.setToolTip(item.data_obj.name)
            self._save_state()

    def get_cell_bounds(self, cell_name):
        cell = self.project.cells.get(cell_name)
        if not cell:
            return QRectF()

        total_rect = QRectF()

        for p_data in cell.polygons:
            for pt in p_data.points:
                total_rect = total_rect.united(QRectF(QPointF(pt[0], pt[1]), QSize(0,0)))

        for e_data in cell.ellipses:
            total_rect = total_rect.united(QRectF(*e_data.rect))

        for ref in cell.references:
            child_bounds = self.get_cell_bounds(ref.cell)
            if not child_bounds.isNull():
                transform = QTransform()
                transform.translate(ref.origin[0], ref.origin[1])
                transform.rotate(ref.rotation)
                transform.scale(ref.magnification, ref.magnification)

                transformed_child_bounds = transform.mapRect(child_bounds)
                total_rect = total_rect.united(transformed_child_bounds)

        return total_rect

# -------------------------- main --------------------------
def main():
    app = QApplication(sys.argv)
    grid_dialog = GridDialog()
    if grid_dialog.exec_() == QDialog.Accepted:
        spacing, w, h = grid_dialog.get_values()
    else:
        spacing, w, h = 50, 2000, 1500

    win = MainWindow(grid_pitch=spacing, canvas_width=w, canvas_height=h)
    win.resize(1600, 1000)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
