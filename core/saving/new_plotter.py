from dataclasses import dataclass, field

from typing import Optional

import numpy as np



@ dataclass
class TimeSeries:
    x: np.ndarray
    y: np.ndarray
    name: Optional[str] = None
    color: Optional[str] = None
    linewidth: Optional[float] = None
    highlight: bool = False
    
    @classmethod
    def create(
        self,
        x: np.ndarray,
        y: np.ndarray = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
        linewidth: Optional[float] = None,
        highlight: bool = False,
    ):
        return self(
            x = self._regulate_x_y_input(x),
            y = self._regulate_x_y_input(y),
            name = name,
            color = color,
            linewidth = linewidth,
            highlight = highlight,
        )
    
    def _regulate_x_y_input(self, input) -> np.ndarray:
        if input is None:
            return None
        return np.asarray(input)

@dataclass
class Subplot:
    series: list[TimeSeries] = field(default_factory=list)
    name: Optional[str]
    x_label: Optional[str] = ""
    y_label: Optional[str] = ""
    
    @classmethod
    def create(
        self,
        series: list[TimeSeries],
        name: Optional[str] = None,
        x_label: Optional[str] = "",
        y_label: Optional[str] = "",
    ):
        return self(
            series = series,
            name = name,
            x_label = x_label,
            y_label = y_label,
        )
    
    def add_series(self, time_series:TimeSeries) -> None:
        self.series.append(time_series)
        return

@dataclass
class Page:
    subplots: list[Subplot] = field(default_factory=list)
    name: Optional[str] = None
    
    @classmethod
    def create(
        self,
        subplots: list[Subplot],
        name: Optional[str] = None,
    ):
        return self(
            subplots = subplots,
            name = name,
        )
    
    def add_subplot(self, subplot:Subplot) -> None:
        self.subplots.append(subplot)
        return



class TimeSeriesPlotter():
    def __init__(self):
        # plot preset
        # data accumulator
        # model plotter (plotter specific for plotting model result of a dataloader)
        # png plotter
        # pdf plotter
        return
    
    def insert_time_series(self, time_series):
        return
    
    def save_to_png():
        return
    
    def save_to_pdf():
        return
    


class PlotSetting():
    
    ### figure
    figsize: tuple = (10, 8)
    dpi: int = 200

    ### lines
    linewidth: float = 1
    highlight_linewidth: float = 5
    
    ### axis ticks
    x_axis_tick_size: int = 8
    y_axis_tick_size: int = 8
    x_axis_pad: int = 5
    y_axis_pad: int = 5
    
    ### color palettes
    color: list = [
        '#8B0000',  # DarkRed
        '#006400',  # DarkGreen
        '#000080',  # Navy
        '#4B0082',  # Indigo
        '#D2691E',  # Chocolate
        '#2F4F4F',  # DarkSlateGray
        '#9400D3',  # DarkViolet
        '#191970',  # MidnightBlue
        '#B22222',  # FireBrick
        '#A0522D',  # Sienna
        '#556B2F',  # DarkOliveGreen
        '#0000CD',  # MediumBlue
    ]
    
    def as_dict(self) -> dict:
        return {
            "figsize": self.figsize,
            "dpi": self.dpi,
            "linewidth": self.linewidth,
            "highlight_linewidth": self.highlight_linewidth,
            "x_axis_tick_size": self.x_axis_tick_size,
            "y_axis_tick_size": self.y_axis_tick_size,
            "x_axis_pad": self.x_axis_pad,
            "y_axis_pad": self.y_axis_pad,
            "color": self.color,
        }
    
    

class DataAccumulator():
    def __init__(self):
        ### accumulate per page
        self.pages: list[Page] = []
        return
    
    ###
    
    def add_page(self, subplots:list[Subplot]):
        return
    
    ###


class PNG_Plotter():
    pass


class PDF_Plotter():
    pass