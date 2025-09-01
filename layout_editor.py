import json
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple

from PyQt5.QtCore import Qt, QRectF, QPointF, QSize
from PyQt5.QtGui import QBrush, QPen, QColor, QPolygonF, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QColorDialog,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPolygonItem,
    QGraphicsItem, QToolBar, QComboBox, QPushButton, QLabel, QLineEdit,
    QSpinBox, QCheckBox, QDialog, QVBoxLayout, QDialogButtonBox, QToolTip
)
from PyQt5.QtSvg import QSvgGenerator

# -------------------------- Data model --------------------------

@dataclass
class Layer:
    name: str
    color: Tuple[int, int, int]

@dataclass
class Shape:
    kind: str
    layer: str
    rect: Tuple[float, float, float, float] = None
    points: List[Tuple[float, float]] = None

# -------------------------- Grid dialog --------------------------

class GridDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grid Setup")
        layout = QVBoxLayout()
        self.spacing_input = QLineEdit("50")
        self.size_input_w = QLineEdit("2000")
        self.size_input_h = QLineEdit("2000")
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
        return int(self.spacing_input.text()), int(self.size_input_w.text()), int(self.size_input_h.text())

# -------------------------- Graphics items --------------------------

class RectItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, layer: Layer):
        super().__init__(rect)
        self.layer = layer
        self.base_color = QColor(*self.layer.color)
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
        self.refresh_appearance(selected=False)

    def refresh_appearance(self, selected=False):
        color = self.base_color.lighter(130) if selected else self.base_color
        self.setBrush(QBrush(color, Qt.SolidPattern))
        self.setPen(QPen(Qt.black, 0.5))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self.refresh_appearance(selected=bool(value))
        return super().itemChange(change, value)


class PolyItem(QGraphicsPolygonItem):
    def __init__(self, poly: QPolygonF, layer: Layer):
        super().__init__(poly)
        self.layer = layer
        self.base_color = QColor(*self.layer.color)
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
        self.refresh_appearance(selected=False)

    def refresh_appearance(self, selected=False):
        color = self.base_color.lighter(130) if selected else self.base_color
        self.setBrush(QBrush(color, Qt.SolidPattern))
        self.setPen(QPen(Qt.black, 0.5))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self.refresh_appearance(selected=bool(value))
        return super().itemChange(change, value)

# -------------------------- View/Scene --------------------------

class Canvas(QGraphicsView):
    MODE_SELECT = "select"
    MODE_RECT = "rect"
    MODE_POLY = "poly"
    MODE_MOVE = "move"

    def __init__(self, scene: QGraphicsScene, get_active_layer, parent=None):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        self.mode = self.MODE_SELECT
        self.start_pos = None
        self.temp_rect = None
        self.temp_poly_points: List[QPointF] = []
        self.get_active_layer = get_active_layer
        self._panning = False
        self._pan_start = None
        self.parent_win = parent
        self.setDragMode(QGraphicsView.RubberBandDrag)
        # Customize selection rectangle to be bright red
        self.setStyleSheet("QGraphicsView::rubberBand { border: 2px solid red; }")

    def set_mode(self, mode: str):
        self.mode = mode
        for item in self.scene().items():
            if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem)):
                if mode == self.MODE_MOVE:
                    item.setFlag(QGraphicsItem.ItemIsMovable, True)
                else:
                    item.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.temp_cancel()

    def temp_cancel(self):
        if self.temp_rect:
            self.scene().removeItem(self.temp_rect)
            self.temp_rect = None
        self.temp_poly_points.clear()
        self.viewport().setCursor(Qt.ArrowCursor)

    def snap_point(self, p: QPointF) -> QPointF:
        if not self.parent_win.chk_snap.isChecked():
            return p
        pitch = self.parent_win.spin_grid.value()
        x = round(p.x() / pitch) * pitch
        y = round(p.y() / pitch) * pitch
        return QPointF(x, y)

    # ----- mouse -----
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self.mode == self.MODE_RECT and event.button() == Qt.LeftButton:
            self.start_pos = self.snap_point(self.mapToScene(event.pos()))
            self.temp_rect = QGraphicsRectItem(QRectF(self.start_pos, self.start_pos))
            self.temp_rect.setPen(QPen(Qt.gray, 0.6, Qt.DashLine))
            self.scene().addItem(self.temp_rect)
            event.accept()
            return
        if self.mode == self.MODE_POLY and event.button() == Qt.LeftButton:
            p = self.snap_point(self.mapToScene(event.pos()))
            self.temp_poly_points.append(p)
            self._draw_temp_poly()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        lm = self.parent_win.left_margin
        bm = self.parent_win.bottom_margin
        grid_w = self.parent_win.grid_width
        grid_h = self.parent_win.grid_height
        pitch = self.parent_win.spin_grid.value()

        # --- Tooltip for grid cell coordinates ---
        x = scene_pos.x() - lm
        y = grid_h - scene_pos.y()  # invert Y to match bottom-left origin

        if 0 <= x <= grid_w and 0 <= y <= grid_h:
            cell_x = int(x // pitch) * pitch
            cell_y = int(y // pitch) * pitch
            QToolTip.showText(event.globalPos(), f"({cell_x}, {cell_y})")
        else:
            QToolTip.hideText()

        # --- Panning with middle mouse ---
        if self._panning and self._pan_start is not None:
            delta = self._pan_start - event.pos()
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
            event.accept()
            return

        # --- Rectangle drawing ---
        if self.mode == self.MODE_RECT and self.temp_rect and self.start_pos:
            cur = self.snap_point(scene_pos)
            rect = QRectF(self.start_pos, cur).normalized()
            self.temp_rect.setRect(rect)
            event.accept()
            return

        super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        snapped = self.snap_point(scene_pos)

        # --- End panning ---
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        # --- Rectangle drawing mode: finalize with black outline ---
        if self.mode == self.MODE_RECT and event.button() == Qt.LeftButton and self.start_pos:
            # remove preview if any
            if getattr(self, "temp_rect", None) is not None:
                if self.temp_rect.scene():
                    self.scene().removeItem(self.temp_rect)
                self.temp_rect = None

            rect = QRectF(self.start_pos, snapped).normalized()
            layer = self.get_active_layer()
            if rect.width() > 1 and rect.height() > 1 and layer:
                item = RectItem(rect, layer)          # RectItem already sets black pen, selectable & movable flags
                self.scene().addItem(item)

            self.start_pos = None
            event.accept()
            return

        # Let default handlers run (selection, moving, etc.)
        super().mouseReleaseEvent(event)

        # --- Move mode: snap selected items to grid on release ---
        if self.mode == self.MODE_MOVE and self.parent_win.chk_snap.isChecked():
            grid = self.parent_win.spin_grid.value()
            lm = self.parent_win.left_margin  # grid x origin starts at left margin

            for item in self.scene().selectedItems():
                if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem)):
                    # current scene top-left of the item’s bounding box
                    tl_scene = item.mapToScene(item.boundingRect().topLeft())

                    # snap x relative to the grid origin at lm
                    snap_x = lm + round((tl_scene.x() - lm) / grid) * grid
                    # snap y to nearest horizontal grid line (origin at 0)
                    snap_y = round(tl_scene.y() / grid) * grid

                    # move by the delta to the snapped point
                    dx = snap_x - tl_scene.x()
                    dy = snap_y - tl_scene.y()
                    if dx or dy:
                        item.moveBy(dx, dy)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.temp_cancel()
            event.accept()
            return
        if event.key() == Qt.Key_Delete:
            for it in self.scene().selectedItems():
                self.scene().removeItem(it)
            event.accept()
            return
        super().keyPressEvent(event)

    # ----- polygon preview -----
    def _draw_temp_poly(self):
        for it in getattr(self, "_temp_preview_items", []):
            self.scene().removeItem(it)
        self._temp_preview_items = []
        n = len(self.temp_poly_points)
        if n >= 2:
            poly = QPolygonF(self.temp_poly_points)
            preview = QGraphicsPolygonItem(poly)
            preview.setPen(QPen(Qt.darkGray, 0.6, Qt.DashLine))
            self.scene().addItem(preview)
            self._temp_preview_items.append(preview)

    def finish_polygon(self):
        if len(self.temp_poly_points) >= 3:
            layer = self.get_active_layer()
            poly = QPolygonF(self.temp_poly_points)
            item = PolyItem(poly, layer)
            self.scene().addItem(item)
        self.temp_cancel()

# -------------------------- Main window --------------------------

class MainWindow(QMainWindow):
    def __init__(self, grid_pitch=50, canvas_width=2000, canvas_height=1500):
        super().__init__()
        self.setWindowTitle("Mask Layout Editor")
        self.left_margin = 50
        self.bottom_margin = 30
        self.grid_pitch = grid_pitch
        self.grid_width = canvas_width
        self.grid_height = canvas_height
        total_width = self.left_margin + self.grid_width
        total_height = self.grid_height + self.bottom_margin
        self.scene = QGraphicsScene(0, 0, total_width, total_height)
        self.view = Canvas(self.scene, self.active_layer, parent=self)
        self.setCentralWidget(self.view)

        # Layers
        self.layers: List[Layer] = [
            Layer("N-doping", (0, 0, 255)),
            Layer("P-doping", (255, 0, 0)),
            Layer("Oxide", (0, 200, 0)),
            Layer("Metal", (200, 200, 200)),
            Layer("Contact", (255, 200, 0))
        ]
        self.layer_by_name = {l.name: l for l in self.layers}

        self._build_toolbar()
        self._build_actions()
        self.statusBar().showMessage("Ready")
        self._draw_grid(self.grid_pitch)

    # ----- UI -----
    def _build_toolbar(self):
        tb = QToolBar("Tools", self)
        tb.setIconSize(QSize(50, 50))  # larger icons/buttons
        tb.setStyleSheet("""
            QToolBar {
                spacing: 10px;          /* space between buttons */
                padding: 5px;           /* inner padding */
                background: #f0f0f0;    /* toolbar background */
                min-height: 70px;       /* make toolbar taller */
            }
        """)
        self.addToolBar(tb)

        # Default button style (only padding)
        self.default_btn_style = "padding:10px 20px; border-radius:6px;"
        self.active_btn_style = "padding:10px 20px; border-radius:6px; background-color: #ccc;"  # subtle light

        # Store buttons
        self.btn_select = QPushButton("Select")
        self.btn_select.setStyleSheet(self.default_btn_style)
        self.btn_select.clicked.connect(lambda: self.set_active_mode(Canvas.MODE_SELECT))
        tb.addWidget(self.btn_select)

        self.btn_rect = QPushButton("Rect")
        self.btn_rect.setStyleSheet(self.default_btn_style)
        self.btn_rect.clicked.connect(lambda: self.set_active_mode(Canvas.MODE_RECT))
        tb.addWidget(self.btn_rect)

        self.btn_poly = QPushButton("Poly")
        self.btn_poly.setStyleSheet(self.default_btn_style)
        self.btn_poly.clicked.connect(lambda: self.set_active_mode(Canvas.MODE_POLY))
        tb.addWidget(self.btn_poly)

        self.btn_move = QPushButton("Move")
        self.btn_move.setStyleSheet(self.default_btn_style)
        self.btn_move.clicked.connect(lambda: self.set_active_mode(Canvas.MODE_MOVE))
        tb.addWidget(self.btn_move)

        tb.addSeparator()
        tb.addWidget(QLabel("Layer:"))
        self.cbo_layer = QComboBox()
        for l in self.layers:
            self.cbo_layer.addItem(l.name)
        tb.addWidget(self.cbo_layer)

        btn_color = QPushButton("Color")
        btn_color.setStyleSheet("padding:5px 10px;")
        btn_color.clicked.connect(self.pick_layer_color)
        tb.addWidget(btn_color)

        tb.addSeparator()
        btn_finish_poly = QPushButton("Finish Poly")
        btn_finish_poly.setStyleSheet("padding:5px 10px;")
        btn_finish_poly.clicked.connect(self.view.finish_polygon)
        tb.addWidget(btn_finish_poly)

        btn_delete_shape = QPushButton("Delete Shape")
        btn_delete_shape.setStyleSheet("padding:5px 10px;")
        btn_delete_shape.clicked.connect(self.delete_selected_shapes)
        tb.addWidget(btn_delete_shape)

        tb.addSeparator()
        self.chk_snap = QCheckBox("Snap")
        self.chk_snap.setChecked(True)
        tb.addWidget(self.chk_snap)

        tb.addWidget(QLabel("Grid:"))
        self.spin_grid = QSpinBox()
        self.spin_grid.setRange(1, 1000)
        self.spin_grid.setValue(self.grid_pitch)
        tb.addWidget(self.spin_grid)

        btn_update_grid = QPushButton("Update Grid")
        btn_update_grid.setStyleSheet("padding:5px 10px;")
        btn_update_grid.clicked.connect(self.update_grid)
        tb.addWidget(btn_update_grid)

    def set_active_mode(self, mode):
        self.view.set_mode(mode)
        # Highlight only the active button
        self.btn_select.setStyleSheet(self.active_btn_style if mode == Canvas.MODE_SELECT else self.default_btn_style)
        self.btn_rect.setStyleSheet(self.active_btn_style if mode == Canvas.MODE_RECT else self.default_btn_style)
        self.btn_poly.setStyleSheet(self.active_btn_style if mode == Canvas.MODE_POLY else self.default_btn_style)
        self.btn_move.setStyleSheet(self.active_btn_style if mode == Canvas.MODE_MOVE else self.default_btn_style)

    def _build_actions(self):
        menu = self.menuBar()
        filem = menu.addMenu("&File")

        act_new = QAction("New", self)
        act_new.triggered.connect(self.new_doc)
        filem.addAction(act_new)

        act_open = QAction("Open JSON…", self)
        act_open.triggered.connect(self.open_json)
        filem.addAction(act_open)

        act_save = QAction("Save JSON…", self)
        act_save.triggered.connect(self.save_json)
        filem.addAction(act_save)

        filem.addSeparator()
        act_export_svg = QAction("Export SVG…", self)
        act_export_svg.triggered.connect(self.export_svg)
        filem.addAction(act_export_svg)

        filem.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        filem.addAction(act_quit)

    # ----- layer ops -----
    def active_layer(self) -> Layer:
        name = self.cbo_layer.currentText()
        layer = self.layer_by_name.get(name)
        if layer:
            # highlight active layer shapes slightly
            for it in self.scene.items():
                if getattr(it, "layer", None) == layer:
                    color = QColor(*layer.color)
                    highlight = QColor(min(color.red()+40,255),
                                    min(color.green()+40,255),
                                    min(color.blue()+40,255))
                    it.setBrush(QBrush(highlight, Qt.SolidPattern))
                elif hasattr(it, "layer"):
                    it.refresh_appearance()  # reset other layers
        return layer

    def delete_selected_shapes(self):
        selected = self.scene.selectedItems()
        if not selected:
            return
        for item in selected:
            self.scene.removeItem(item)

    def pick_layer_color(self):
        layer = self.active_layer()
        if not layer:
            return
        color = QColorDialog.getColor(QColor(*layer.color), self, "Pick color")
        if not color.isValid():
            return
        layer.color = (color.red(), color.green(), color.blue())
        for it in self.scene.items():
            if getattr(it, "layer", None) == layer:
                it.refresh_appearance()

    # ----- document ops -----
    def new_doc(self):
        self.scene.clear()
        self._draw_grid(self.spin_grid.value())

    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON Files (*.json)")
        if not path:
            return
        data = {
            "layers": [asdict(l) for l in self.layers],
            "shapes": []
        }
        for it in self.scene.items():
            if not hasattr(it, "layer"):
                continue
            if isinstance(it, RectItem):
                r = it.rect()
                data["shapes"].append(Shape(
                    kind="rect",
                    layer=it.layer.name,
                    rect=(r.x(), r.y(), r.width(), r.height())
                ).__dict__)
            elif isinstance(it, PolyItem):
                pts = [(p.x(), p.y()) for p in it.polygon()]
                data["shapes"].append(Shape(
                    kind="poly",
                    layer=it.layer.name,
                    points=pts
                ).__dict__)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def open_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open JSON", "", "JSON Files (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)
        self.scene.clear()
        self._draw_grid(self.spin_grid.value())

        self.layers.clear()
        self.layer_by_name.clear()
        self.cbo_layer.clear()
        for L in data.get("layers", []):
            layer = Layer(L["name"], tuple(L["color"]))
            self.layers.append(layer)
            self.layer_by_name[layer.name] = layer
            self.cbo_layer.addItem(layer.name)
        if self.layers:
            self.cbo_layer.setCurrentText(self.layers[0].name)

        for S in data.get("shapes", []):
            layer = self.layer_by_name.get(S["layer"])
            if not layer:
                continue
            if S["kind"] == "rect" and S.get("rect"):
                x, y, w, h = S["rect"]
                item = RectItem(QRectF(x, y, w, h), layer)
                self.scene.addItem(item)
            elif S["kind"] == "poly" and S.get("points"):
                poly = QPolygonF([QPointF(px, py) for px, py in S["points"]])
                item = PolyItem(poly, layer)
                self.scene.addItem(item)

    def export_svg(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", "", "SVG Files (*.svg)")
        if not path:
            return
        gen = QSvgGenerator()
        gen.setFileName(path)
        rect = self.scene.itemsBoundingRect().adjusted(-50, -50, 50, 50)
        gen.setViewBox(rect)
        painter = QPainter(gen)
        self.scene.render(painter, target=QRectF(), source=rect)
        painter.end()

    # ----- grid -----
    def update_grid(self):
        self._draw_grid(self.spin_grid.value())

    def _draw_grid(self, pitch=50):
        self.scene.clear()
        w = self.grid_width
        h = self.grid_height
        lm = self.left_margin
        bm = self.bottom_margin

        # grid lines
        grid_pen = QPen(QColor(230, 230, 230), 0)
        for x in range(0, w + 1, pitch):
            self.scene.addLine(lm + x, 0, lm + x, h, grid_pen)
        for y in range(0, h + 1, pitch):
            self.scene.addLine(lm, y, lm + w, y, grid_pen)

        # axes
        axis_pen = QPen(QColor(0, 0, 0), 2)
        self.scene.addLine(lm, 0, lm, h, axis_pen)  # left axis
        self.scene.addLine(lm, h, lm + w, h, axis_pen)  # bottom axis

        # axis labels
        for x in range(0, w + 1, pitch):
            text = self.scene.addText(str(x))
            text.setPos(lm + x - 5, h + 2)
        for y in range(0, h + 1, pitch):
            text = self.scene.addText(str(y))
            text.setPos(2, h - y - 10)

# -------------------------- main --------------------------

def main():
    app = QApplication(sys.argv)
    # show grid dialog first
    dlg = GridDialog()
    if dlg.exec_() == QDialog.Accepted:
        spacing, w, h = dlg.get_values()
    else:
        spacing, w, h = 50, 2000, 1500

    win = MainWindow(grid_pitch=spacing, canvas_width=w, canvas_height=h)
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
