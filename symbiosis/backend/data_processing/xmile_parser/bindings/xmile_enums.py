# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XMILE constants."""

import enum


class AccessType(enum.Enum):
  """Used in conjunction with submodels to define input and outputs."""

  INPUT = "input"
  OUTPUT = "output"


class AfterChoices(enum.Enum):
  ONE_TIME = "one_time"
  ONE_DT = "one_dt"


class AliasBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class AliasFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class AliasFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class AliasLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class AliasTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class AuxBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class AuxFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class AuxFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class AuxLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class AuxTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class ButtonBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class ButtonFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class ButtonFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class ButtonLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class ButtonStyle(enum.Enum):
  SQUARE = "square"
  ROUNDED = "rounded"
  CAPSULE = "capsule"


class ButtonTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class ConnectorBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class ConnectorFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class ConnectorFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class ConnectorLineStyle(enum.Enum):
  SOLID = "solid"
  DASHED = "dashed"
  VENDOR_SPECIFIC = "vendor specific"


class ConnectorTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class DataType(enum.Enum):
  CSV = "CSV"
  XLS = "XLS"
  XML = "XML"


class EventSimAction(enum.Enum):
  PAUSE = "pause"
  STOP = "stop"
  MESSAGE = "message"


class FlowBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class FlowFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class FlowFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class FlowLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class FlowTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class FormatDisplayAs(enum.Enum):
  NUMBER = "number"
  CURRENCY = "currency"
  PERCENT = "percent"


class FrequencyType(enum.Enum):
  AUTOMATIC = "automatic"
  ON_DEMAND = "on_demand"


class GfType(enum.Enum):
  CONTINUOUS = "continuous"
  EXTRAPOLATE = "extrapolate"
  DISCRETE = "discrete"


class GraphBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class GraphFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class GraphFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class GraphTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class GraphType(enum.Enum):
  TIME_SERIES = "time_series"
  SCATTER = "scatter"
  BAR = "bar"


class GraphicalInputBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class GraphicalInputFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class GraphicalInputFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class GraphicalInputLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class GraphicalInputTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class GraphicsFrameBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class GraphicsFrameFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class GraphicsFrameFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class GraphicsFrameLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class GraphicsFrameTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class GroupBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class GroupFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class GroupFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class GroupLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class GroupTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class ItemBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class ItemDisplayAs(enum.Enum):
  NUMBER = "number"
  CURRENCY = "currency"
  PERCENT = "percent"


class ItemFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class ItemFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class ItemTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class KnobBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class KnobFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class KnobFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class KnobLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class KnobTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class LinkEffect(enum.Enum):
  NONE = "none"
  DISSOLVE = "dissolve"
  CHECKERBOARD = "checkerboard"
  BARS = "bars"
  WIPE_LEFT = "wipe_left"
  WIPE_RIGHT = "wipe_right"
  WIPE_TOP = "wipe_top"
  WIPE_BOTTOM = "wipe_bottom"
  WIPE_CLOCKWISE = "wipe_clockwise"
  WIPE_COUNTERCLOCKWISE = "wipe_counterclockwise"
  IRIS_IN = "iris_in"
  IRIS_OUT = "iris_out"
  DOORS_CLOSE = "doors_close"
  DOORS_OPEN = "doors_open"
  VENETIAN_LEFT = "venetian_left"
  VENETIAN_RIGHT = "venetian_right"
  VENETIAN_TOP = "venetian_top"
  VENETIAN_BOTTOM = "venetian_bottom"
  PUSH_BOTTOM = "push_bottom"
  PUSH_TOP = "push_top"
  PUSH_LEFT = "push_left"
  PUSH_RIGHT = "push_right"


class ListInputBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class ListInputFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class ListInputFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class ListInputLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class ListInputTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class MacroApplyto(enum.Enum):
  INFLOWS = "inflows"
  OUTFLOWS = "outflows"
  UPSTREAM = "upstream"
  DOWNSTREAM = "downstream"


class MacroFilter(enum.Enum):
  STOCK = "stock"
  FLOW = "flow"


class StockBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class MenuActionChoices(enum.Enum):
  OPEN = "open"
  CLOSE = "close"
  SAVE = "save"
  SAVE_AS = "save_as"
  SAVE_AS_IMAGE = "save_as_image"
  REVERT = "revert"
  PRINT_SETUP = "print_setup"
  PRINT = "print"
  PRINT_SCREEN = "print_screen"
  RUN = "run"
  PAUSE = "pause"
  RESUME = "resume"
  STOP = "stop"
  RUN_RESTORE = "run_restore"
  RESTORE_ALL = "restore_all"
  RESTORE_SLIDERS = "restore_sliders"
  RESTORE_KNOBS = "restore_knobs"
  RESTORE_LIST_INPUTS = "restore_list_inputs"
  RESTORE_GRAPHICAL_INPUTS = "restore_graphical_inputs"
  RESTORE_SWITCHES = "restore_switches"
  RESTORE_NUMERIC_DISPLAYS = "restore_numeric_displays"
  RESTORE_GRAPHS_TABLES = "restore_graphs_tables"
  RESTORE_LAMPS_GAUGES = "restore_lamps_gauges"
  DATA_MANAGER = "data_manager"
  SAVE_DATA_NOW = "save_data_now"
  IMPORT_NOW = "import_now"
  EXPORT_NOW = "export_now"
  EXIT = "exit"
  FIND = "find"
  RUN_SPECS = "run_specs"


class ModuleBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class ModuleFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class ModuleFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class ModuleLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class ModuleTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class NumericDisplayBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class NumericDisplayDisplayAs(enum.Enum):
  NUMBER = "number"
  CURRENCY = "currency"
  PERCENT = "percent"


class NumericDisplayFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class NumericDisplayFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class NumericDisplayLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class NumericDisplayTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class OptionsBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class OptionsFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class OptionsFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class OptionsLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class OptionsLayout(enum.Enum):
  VERTICAL = "vertical"
  HORIZONTAL = "horizontal"


class OptionsTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class Orientation(enum.Enum):
  HORIZONTAL = "horizontal"
  VERTICAL = "vertical"


class OrientationType(enum.Enum):
  VERTICAL = "vertical"
  HORIZONTAL = "horizontal"


class Penstyle(enum.Enum):
  SOLID = "solid"
  DOTTED = "dotted"
  DASHED = "dashed"
  DOT_DASHED = "dot_dashed"


class PopupBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class PopupFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class PopupFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class PopupLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class PopupTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class ReportBalances(enum.Enum):
  BEGINNING = "beginning"
  ENDING = "ending"


class ReportFlows(enum.Enum):
  INSTANTANEOUS = "instantaneous"
  SUMMED = "summed"


class ReportIntervalValue(enum.Enum):
  DT = "DT"


class RunBy(enum.Enum):
  ALL = "all"
  GROUP = "group"
  MODULE = "module"


class ShapeType(enum.Enum):
  RECTANGLE = "rectangle"
  CIRCLE = "circle"
  NAME_ONLY = "name_only"


class SliderBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class SliderFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class SliderFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class StandardTimeUnitType(enum.Enum):
  DAY = "day"
  HOUR = "hour"
  MICROSECOND = "microsecond"
  MILLISECOND = "millisecond"
  MINUTE = "minute"
  MONTH = "month"
  NANOSECOND = "nanosecond"
  QUARTER = "quarter"
  SECOND = "second"
  WEEK = "week"
  YEAR = "year"


class SliderLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class SliderTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class StandardBorderWidthType(enum.Enum):
  THIN = "thin"
  THICK = "thick"


class StandardColorType(enum.Enum):
  AQUA = "aqua"
  BLACK = "black"
  BLUE = "blue"
  FUCHSIA = "fuchsia"
  GRAY = "gray"
  GREEN = "green"
  LIME = "lime"
  MAROON = "maroon"
  NAVY = "navy"
  OLIVE = "olive"
  PURPLE = "purple"
  RED = "red"
  SILVER = "silver"
  TEAL = "teal"
  WHITE = "white"
  YELLOW = "yellow"


class StandardMethodType(enum.Enum):
  EULER = "euler"
  RK2 = "rk2"
  RK2_AUTO = "rk2_auto"
  RK4 = "rk4"
  RK4_AUTO = "rk4_auto"


class StockFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class StockFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class StockLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class StockTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class StyleBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class StyleFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class StyleFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class StyleTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class StyleTypeBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class StyleTypeFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class StyleTypeFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class StyleTypeLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class StyleTypeTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class SwitchBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class SwitchFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class SwitchFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class SwitchLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class SwitchSwitchStyle(enum.Enum):
  TOGGLE = "toggle"
  PUSH_BUTTON = "push_button"


class SwitchTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class TableBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class TableFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class TableFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class TableHeaderBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class TableHeaderFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class TableHeaderFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class TableHeaderTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class TableItemType(enum.Enum):
  VARIABLE = "variable"
  SPACE = "space"
  TIME = "time"


class TableTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class TextAlign(enum.Enum):
  LEFT = "left"
  RIGHT = "right"
  CENTER = "center"


class TextBoxBorderStyle(enum.Enum):
  NONE = "none"
  SOLID = "solid"


class TextBoxFontStyle(enum.Enum):
  NORMAL = "normal"
  ITALIC = "italic"


class TextBoxFontWeight(enum.Enum):
  NORMAL = "normal"
  BOLD = "bold"


class TextBoxLabelSide(enum.Enum):
  TOP = "top"
  LEFT = "left"
  CENTER = "center"
  BOTTOM = "bottom"
  RIGHT = "right"


class TextBoxTextDecoration(enum.Enum):
  NORMAL = "normal"
  UNDERLINE = "underline"


class ThresholdDirection(enum.Enum):
  INCREASING = "increasing"
  DECREASING = "decreasing"


class ThresholdRepeat(enum.Enum):
  EACH = "each"
  ONCE = "once"
  ONCE_EVER = "once_ever"


class TransparencyType(enum.Enum):
  OPAQUE = "opaque"
  TRANSPARENT = "transparent"


class UsesArraysInvalidIndexValue(enum.Enum):
  VALUE_0 = "0"
  NA_N = float("nan")


class VerticalTextAlign(enum.Enum):
  TOP = "top"
  BOTTOM = "bottom"
  CENTER = "center"


class ViewPageOrientation(enum.Enum):
  LANDSCAPE = "landscape"
  PORTRAIT = "portrait"


class ViewPageSequence(enum.Enum):
  ROW = "row"
  COLUMN = "column"


class ViewType(enum.Enum):
  STOCK_FLOW = "stock_flow"
  INTERFACE = "interface"
  POPUP = "popup"
