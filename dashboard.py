# This is a simplified dashboard to visualize the MSNA signal and the predicted bursts.
# It is built with Bokeh and can be run with argparse.
#
import pandas as pd
import numpy as np
import argparse
import os

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import RangeTool
from bokeh.models import Range1d
from bokeh.models import Legend
from bokeh.layouts import column
from bokeh.palettes import Category10
from bokeh.models.tools import HoverTool
from bokeh.io import save
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from tornado.ioloop import IOLoop

from msna_detect.filters import normalize_msna

# Global plot parameters
MAIN_PLOT_WIDTH = 1500
MAIN_PLOT_HEIGHT = 400
RANGE_PLOT_WIDTH = 1500
RANGE_PLOT_HEIGHT = 150
Y_RANGE_START = -2.0
Y_RANGE_END = 5.0
TRUE_BURST_Y_POS = 4.5
PRED_BURST_Y_POS = 4.8
PROBABILITY_OFFSET = -1.8


def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ["Integrated MSNA", "Burst", "Predicted Burst", "Predicted Probability"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Normalize data
    df["Integrated MSNA"] = normalize_msna(df["Integrated MSNA"].to_numpy()) / 2
    df["Predicted Probability"] = df["Predicted Probability"].to_numpy() + PROBABILITY_OFFSET
    
    # Create time index if not present
    if "time" not in df.columns:
        df["time"] = np.arange(len(df))
    
    return df


def create_data_sources(df):
    """Create Bokeh data sources from the dataframe"""
    # Main data source for time series data
    data_source = ColumnDataSource(
        data = dict(
            time = df["time"].values,
            integrated_msna = df["Integrated MSNA"].values,
            burst = df["Burst"].values,
            predicted_burst = df["Predicted Burst"].values,
            predicted_probability = df["Predicted Probability"].values
        )
    )
    
    # Extract burst positions for scatter plots
    true_burst_mask = df["Burst"].astype(bool)
    pred_burst_mask = df["Predicted Burst"].astype(bool)
    
    # Get indices for burst markers
    true_burst_indices = df.loc[true_burst_mask, "time"].values
    pred_burst_indices = df.loc[pred_burst_mask, "time"].values
    
    # Set fixed y-positions for burst markers
    true_burst_y_pos = np.full(len(true_burst_indices), TRUE_BURST_Y_POS)
    pred_burst_y_pos = np.full(len(pred_burst_indices), PRED_BURST_Y_POS)
    
    # Separate data sources for burst markers
    true_burst_source = ColumnDataSource(
        data = dict(time = true_burst_indices, y_pos = true_burst_y_pos)
    )
    pred_burst_source = ColumnDataSource(
        data = dict(time = pred_burst_indices, y_pos = pred_burst_y_pos)
    )
    
    return data_source, true_burst_source, pred_burst_source


def create_plots(data_source, true_burst_source, pred_burst_source, df):
    """Create the main and range plots"""
    # Main plot setup
    main_plot = figure(
        width = MAIN_PLOT_WIDTH, 
        height = MAIN_PLOT_HEIGHT,
        title = "MSNA Signal Analysis",
        x_axis_label = "Time (samples)",
        tools = "pan,box_zoom,wheel_zoom,reset,save",
        output_backend = "webgl"
    )
    
    # Set fixed y-range and disable y-axis interaction
    main_plot.y_range = Range1d(start = Y_RANGE_START, end = Y_RANGE_END, bounds = (Y_RANGE_START, Y_RANGE_END))
    main_plot.yaxis.visible = False
    
    # Set initial x-range
    max_time = df["time"].max()
    initial_end = min(10000, max_time)
    main_plot.x_range.start = 0
    main_plot.x_range.end = initial_end
    
    # Plot the integrated MSNA signal
    msna_line = main_plot.line(
        "time", "integrated_msna", 
        source = data_source,
        line_width = 1, 
        color = Category10[10][0], 
        alpha = 0.8,
    )
    
    # Plot the predicted probability (now on same axis with offset)
    prob_line = main_plot.line(
        "time", "predicted_probability", 
        source = data_source,
        line_width = 2, 
        color = Category10[10][1], 
        alpha = 0.7,
    )
    
    # True burst markers (circles)
    true_bursts = main_plot.scatter(
        "time", "y_pos",
        source = true_burst_source,
        size = 8,
        marker = "circle",
        color = Category10[10][2],
        alpha = 0.8,
    )
    
    # Predicted burst markers (triangles)
    pred_bursts = main_plot.scatter(
        "time", "y_pos",
        source = pred_burst_source,
        size = 10,
        marker = "triangle",
        color = Category10[10][3],
        alpha = 0.8,
    )
    
    # Add hover tools
    hover_msna = HoverTool(
        tooltips = [("Time", "@time"), ("MSNA", "@integrated_msna{0.000}")],
        renderers = [msna_line]
    )
    hover_prob = HoverTool(
        tooltips = [("Time", "@time"), ("Probability", "@predicted_probability{0.000}")],
        renderers = [prob_line]
    )
    main_plot.add_tools(hover_msna, hover_prob)
    
    # Range tool plot (overview)
    range_plot = figure(
        width = RANGE_PLOT_WIDTH, 
        height = RANGE_PLOT_HEIGHT,
        x_axis_label = "Time (samples)",
        title = "Navigation (drag to pan, resize selection)",
        tools = "pan,reset"
    )
    
    # Plot overview data in range tool
    range_plot.line(
        "time", "integrated_msna", 
        source = data_source,
        line_width = 1, 
        color = Category10[10][0], 
        alpha = 0.6
    )
    
    # Set range plot ranges
    range_plot.x_range.start = 0
    range_plot.x_range.end = max_time
    range_plot.y_range = Range1d(start = Y_RANGE_START, end = Y_RANGE_END)
    range_plot.yaxis.visible = False
    range_plot.xaxis.visible = False
    
    # Create range tool
    range_tool = RangeTool(x_range = main_plot.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    range_plot.add_tools(range_tool)

    # Add legend
    legend = Legend(items = [
        ("Integrated MSNA", [msna_line]),
        ("Predicted Probability", [prob_line]),
        ("True Bursts", [true_bursts]),
        ("Predicted Bursts", [pred_bursts])
    ])
    legend.click_policy = "hide"
    legend.orientation = "horizontal"
    main_plot.add_layout(legend, "above")
    
    return main_plot, range_plot


def run_dashboard(doc, input_filepath: str, save_html: bool = False):
    """Main function to create and run the dashboard"""
    try:
        # Load data
        df = load_data(input_filepath)
        
        # Create data sources
        data_source, true_burst_source, pred_burst_source = create_data_sources(df)
        
        # Create plots
        main_plot, range_plot = create_plots(data_source, true_burst_source, pred_burst_source, df)
        
        # Create layout
        layout = column(
            main_plot,
            range_plot,
            margin = (25, 0, 0, 25)  # (top, right, bottom, left) in pixels
        )
        
        # Add to document
        doc.add_root(layout)
        doc.title = "MSNA Signal Visualization"
        
        # Save to HTML if requested
        if save_html:
            output_path = os.path.splitext(input_filepath)[0] + "_dashboard.html"
            save(layout, filename = output_path)
            print(f"Dashboard saved to: {output_path}")
            
    except Exception as e:
        print(f"Error: Failed to load data: {str(e)}")
        doc.title = "MSNA Signal Visualization - Error"


def main(input_filepath: str, save: bool = False):
    """Main function to start the Bokeh server with the dashboard"""
    bokeh_app = Application(FunctionHandler(lambda doc: run_dashboard(doc, input_filepath, save_html = save)))
    server = Server(
        applications = {"/": bokeh_app}, 
        io_loop = IOLoop(), 
        allow_websocket_origin = ["localhost:5006"]
    )
    server.start()
    print(f"Dashboard running at: http://localhost:5006/")
    print(f"Loading data from: {input_filepath}")
    if save:
        print("HTML file will be saved when dashboard loads.")
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description = "MSNA Signal Visualization Dashboard")
    parser.add_argument(
        "-i", "--input", type = str, required = True, 
        help = "The path to the input data (csv file)."
    )
    parser.add_argument(
        "--save", action = "store_true", default = False,
        help = "Whether to save the dashboard to an HTML file."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(input_filepath = args.input, save = args.save)


