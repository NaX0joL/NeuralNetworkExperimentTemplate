#!!! need fix later asap

import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os
import time
from collections.abc import Iterable

from core.datasets.data_module import DataModule



class Plotter():
    """
    > class to hold plot data that could be printed directly, ported to png, or ported to pdf
    > each plot could hold an arbitrary amount of time series input vertically
    """
    
    def __init__(self, name="plotty"):
        # name for fun
        self.name = name
        
        # param for plot styling
        self.param = self._get_customization_param()
        
        # data to plot, list of tuples:(value, highlight, color, title)
        self.plot_data = []
    
    def who_are_you(self):
        print(f"Hi, I am {self.name}!")
        return self.name
    
    ##########################################################################################################################
    
    ##########################################################################################################################
    
    def _get_customization_param(self):
        param = {}
        
        # basic settings
        figsize = (10, 8) 
        dpi = 200
        new_element = {
            "figsize": figsize,
            "dpi": dpi,
        }
        param.update(new_element)
        
        # line width
        linewidth = 1
        highlight_linewidth = 5
        new_element = {
            "linewidth": linewidth,
            "highlight_linewidth": highlight_linewidth,
        }
        param.update(new_element)
        
        # axis ticks
        x_axis_tick_size = 8
        y_axis_tick_size = 8
        x_axis_pad = 5
        y_axis_pad = 5
        new_element = {
            "x_axis_tick_size": x_axis_tick_size,
            "y_axis_tick_size": y_axis_tick_size,
            "x_axis_pad": x_axis_pad,
            "y_axis_pad": y_axis_pad,
        }
        param.update(new_element)
        
        # color setting
        color = [
            '#8B0000',  # DarkRed
            '#006400',  # DarkGreen
            '#000080',  # Navy (Dark Blue)
            '#4B0082',  # Indigo
            '#D2691E',  # Chocolate (a dark brown/orange)
            '#2F4F4F',  # DarkSlateGray
            '#9400D3',  # DarkViolet
            '#191970',  # MidnightBlue
            '#B22222',  # FireBrick
            '#A0522D',  # Sienna
            '#556B2F',  # DarkOliveGreen
            '#0000CD',  # MediumBlue
        ]
        new_element = {'color': color}
        param.update(new_element)
        
        return param
    
    def _apply_ax_extra_customization(self, ax):
        # remove x axis label
        ax.set_xlabel("", fontsize=0, labelpad=0)
        # remove y axis label
        ax.set_ylabel("", fontsize=0, labelpad=0)
        # customize x-axis numbers
        ax.tick_params(axis='x', labelsize=self.param['x_axis_tick_size'], pad=self.param['x_axis_pad'])
        # customize y-axis numbers
        ax.tick_params(axis='y', labelsize=self.param['x_axis_tick_size'], pad=self.param['y_axis_pad'])
        
        return ax
    
    ##########################################################################################################################
    
    def _create_figure(self, value, highlight, color, title):
        # define the fig and ax to plot
        row = len(value)
        column = 1
        fig, ax = plt.subplots(row, column, sharex=True, figsize=self.param['figsize'], dpi=self.param['dpi'])
        if not isinstance(ax, Iterable): ax = [ax]      # ax is not initiated as list when row==1
        
        # define the horizontal time step based on the longest time series len
        time_step = np.arange(max([len(x) for x in value]))
        
        # plot data one by one
        for index in range(row):
            # actual plotting method
            ax[index].plot(
                time_step[:len(value[index])], 
                value[index], 
                linewidth=self.param['linewidth'],
                color=color[index],
            )
            
            # highlight specified time steps
            if highlight[index] is not None:
                ax[index].plot(
                    time_step[:len(highlight[index])],
                    highlight[index],
                    linewidth=self.param['highlight_linewidth'],
                    color=color[index],
                )
            
            # change the individual plot title
            ax[index].set_title(title[index])
            
            # extra plot customization
            ax[index] = self._apply_ax_extra_customization(ax[index])
        
        return fig
    
    def _preview_plot(self, value, highlight, color, title, show_time=None):
        # clear all figs first
        plt.close('all')
        
        # create matlplotlib fig to show
        fig = self._create_figure(value, highlight, color, title)
        
        plt.tight_layout()
        
        # show plot, with time limit if specified
        if show_time is not None:
            try:
                plt.show(block=False)
                plt.pause(show_time)
                
            except KeyboardInterrupt:
                print("keyboard interrupt, plotting halted")
                plt.close(fig)
                return
        else:
            plt.show()
        
        # clear after
        plt.close(fig)
        
        return
    
    ##########################################################################################################################
    
    def _check_input_type(self, *input):
        # check for individual item in input
        for item in input:
            # time series data only check
            if not isinstance(item, dict):
                # only accept dataframe or tensor
                if not (isinstance(item, pd.DataFrame) or isinstance(item, pd.Series) or isinstance(item, torch.Tensor)):
                    return False
            # time series data with customization check
            else:
                # requires 'value' as a key
                if 'value' in item.keys():
                    # only accept dataframe or tensor
                    if not (isinstance(item['value'], pd.DataFrame) or isinstance(item, pd.Series) or isinstance(item['value'], torch.Tensor)):
                        return False
                else:
                    return False
        return True
    
    def _regularize_data_type(self, input):
        if isinstance(input, torch.Tensor): input = pd.DataFrame(input.cpu().detach().numpy())
        return input
    
    def _check_input_shape(self, input):
        return
    
    def _unpack_data(self, *data):
        optional_keys = ['value', 'highlight', 'color', 'title',]
        
        # init dict to store data separately
        separated_data = {}
        for key in optional_keys:
            separated_data[key] = []
        
        for index, item in enumerate(data):
            if isinstance(item, dict):
                # set default value if optional is missing
                value = item['value']   # mandatory key, no need for checking
                highlight = item['highlight'] if 'highlight' in item.keys() else None
                color = item['color'] if 'color' in item.keys() else self.param['color'][index % len(self.param['color'])]
                title = item['title'] if 'title' in item.keys() else None
                
                separated_data['value'].append(self._regularize_data_type(value))
                separated_data['highlight'].append(self._regularize_data_type(highlight))
                separated_data['color'].append(color)
                separated_data['title'].append(title)
            else:
                separated_data['value'].append(self._regularize_data_type(item))
                separated_data['highlight'].append(None)
                separated_data['color'].append(self.param['color'][index % len(self.param['color'])])
                separated_data['title'].append(None)
        
        return separated_data['value'], separated_data['highlight'], separated_data['color'], separated_data['title']
    
    ##########################################################################################################################
    
    ##########################################################################################################################
    
    def insert_data(self, *data, preview_plot=False, show_time=None):
        """
        accepts arbitrary input to be put in a single plot
        each item in *data could be either
        - single dataframe or tensor
        - a dict with keys:
            'value':        single dataframe or tensor  
            'highlight':    <optional> dataframe containing which time step is highlighted
            'color':        <optional> str of the plot color 
            'title':        <optional> str of the this plot title
        """
        # first check that the input is valid
        assert self._check_input_type(*data), "invalid input type!"
        
        # unpack data into constituent
        value, highlight, color, title = self._unpack_data(*data)
        
        # insert constituent as a tuple into global list
        self.plot_data.append((value, highlight, color, title))
        
        # show plot when inserting
        if preview_plot:
            self._preview_plot(value, highlight, color, title, show_time=show_time)
        
        return
    
    def show_plot(self, show_time=False):
        try:
            for item in self.plot_data:
                self._preview_plot(*item, show_time=show_time)
        except KeyboardInterrupt:
                print("keyboard interrupt, plotting halted")
                return
        return
    
    def port_to_png(self, folder_name='png', singular_png_filename='fig.png', save_path=''):
        # ignore when plot data is empty
        if len(self.plot_data) < 1:
            return
        
        # no need for folder when saving one plot
        if len(self.plot_data) == 1:
            #print(self.plot_data[0])
            fig = self._create_figure(*self.plot_data[0])
            png_save_path = os.path.join(save_path, singular_png_filename)
            fig.savefig(png_save_path, dpi=self.param['dpi'])
            plt.close(fig)
            return
        
        # create folder to house multiple pngs
        dir_path = os.path.join(save_path, folder_name)
        os.makedirs(dir_path, exist_ok=True)
        
        # clear all lingering figs first
        plt.close('all')
        
        for index, item in enumerate(self.plot_data):
            # create figure
            fig = self._create_figure(*item)
            
            # save plot as png
            png_save_path = os.path.join(dir_path, f'fig_{index+1}.png')
            fig.savefig(png_save_path, dpi=self.param['dpi'])
        
            # close fig after use
            plt.close(fig)
        
        return
    
    def port_to_pdf(self, file_name, save_path=''):
        # clear all lingering figs first
        plt.close('all')
        
        pdf_save_path = os.path.join(save_path, file_name)
        with PdfPages(pdf_save_path) as pdf:
            for index, item in enumerate(self.plot_data):
                # create figure
                fig = self._create_figure(*item)
                
                # save plot in pdf page
                pdf.savefig(fig)
                
                # close fig after use
                plt.close(fig) 
        
        return



class ModelPlotter():
    # save model here and plot its output here
    
    def __init__(self, model:nn.Module, data_module:DataModule):
        self.model = model
        self.data_module = data_module
        
        self.plotted_attributes = [
            'value', 'ground_truth',
        ]
        
        return



class PlotSetting():
    # move all the plot visual settings here 
    pass



def plot_Dataframe(*data, show_plot=True, savefig=False, plot_name='unnamed_fig.png', save_path='', show_time=None):
    """
    A smaller and quicker function to plot a bunch of time series dataframe in a big single plot
    """
    plotter = Plotter()
    
    plotter.insert_data(*data, preview_plot=show_plot, show_time=show_time)
    
    if savefig:
        plotter.port_to_png(singular_png_filename=plot_name, save_path=save_path)
        
    return



if __name__ == "__main__":
    pass