"""
This module contains functions for (pre)processing D8 flow direction grids and snapping sample sites to drainage networks.
"""

from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

import funmixer.flow_acc_cfuncs as cf


def read_geo_file(filename: str) -> Tuple[np.ndarray, gdal.Dataset]:
    """Reads a geospatial file"""
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds


def write_geotiff(filename: str, arr: np.ndarray, ds: gdal.Dataset) -> None:
    """Writes a numpy array to a geotiff"""
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(ds.GetProjection())
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)


def check_d8(
    flowdirs_filename: str,
) -> None:
    """
    Checks if a D8 flow direction grid is valid. A valid D8 flow direction grid has the following properties:
    - All boundary nodes are 0
    - All values are 0, 1, 2, 4, 8, 16, 32, 64, 128

    Args:
        flowdirs_filename (str): The filename of the D8 flow direction grid
    """
    print("-" * 50)
    print("Checking D8 flow direction grid at", flowdirs_filename)
    arr, _ = read_geo_file(flowdirs_filename)
    # Cast to int
    arr = arr.astype(int)
    # Check that the only values present are 0, 1, 2, 4, 8, 16, 32, 64, 128 using sets
    unique_values = set(np.unique(arr))
    values_are_valid = unique_values == {0, 1, 2, 4, 8, 16, 32, 64, 128}
    if not values_are_valid:
        print(
            f"VALUE CHECK RESULT: Fail. Invalid values present in D8 flow direction grid: {unique_values}. \n Expected values are { {0, 1, 2, 4, 8, 16, 32, 64, 128} }"
        )
    else:
        print("VALUE CHECK RESULT: Pass.")
    # Check that the boundaries are all 0
    boundaries_are_zero = (
        np.all(arr[0, :] == 0)
        and np.all(arr[-1, :] == 0)
        and np.all(arr[:, 0] == 0)
        and np.all(arr[:, -1] == 0)
    )
    if not boundaries_are_zero:
        print("BOUNDARY CHECK RESULT: Fail. Boundaries of D8 flow direction grid are not all 0.")
    else:
        print("BOUNDARY CHECK RESULT: Pass.")
    # Return True if both boundaries_are_zero and values_are_valid
    result = boundaries_are_zero and values_are_valid
    if result:
        print("D8 flow direction grid is valid.")
    else:
        print("D8 flow direction grid is INVALID.")
    print("-" * 50)


def set_d8_boundaries_to_zero(flowdirs_filename: str) -> None:
    """
    Sets the boundaries of a D8 flow direction grid to zero. Writes the result to a new geotiff with a prefix "zero_bounds_"

    Args:
        flowdirs_filename (str): The filename of the D8 flow direction grid
    """
    arr, ds = read_geo_file(flowdirs_filename)
    arr[0, :] = 0
    arr[-1, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0

    # Save the new file with prefix fixed_bounds_[original_filename].tif, overwriting the original file extension
    name, _ = flowdirs_filename.split(".")
    new_filename = f"zero_bounds_{name}.tif"
    print("Writing new file with zeroed boundaries to", new_filename)
    write_geotiff(new_filename, arr, ds)


def elevation_to_d8(elevation_filename: str) -> None:
    """
    Converts a geospatial raster of elevations to a D8 flow grid and writes it to a file called [filename]_d8.tif.
    Follows convention of ESRI D8 flow directions: right = 1, lower right = 2, bottom = 4, etc.
    All boundary nodes are set to be sinks.

    Args:
        elevation_filename (str): The filename of the elevation raster
    """
    elev, ds = read_geo_file(elevation_filename)
    d8 = cf.topo_to_d8(elev.astype(float))
    # Set boundaries to zero
    d8[0, :] = 0
    d8[-1, :] = 0
    d8[:, 0] = 0
    d8[:, -1] = 0

    name, _ = elevation_filename.split(".")
    new_filename = f"d8_{name}.tif"
    print("Writing valid D8 to", new_filename)
    write_geotiff(new_filename, d8, ds)


def snap_to_drainage(
    flow_dirs_filename: str,
    sample_sites_filename: str,
    drainage_area_threshold: float,
    plot: bool = True,
    save: bool = False,
    nudges: Dict[str, np.ndarray] = {},
):
    """
    Function that snaps sample sites to the nearest drainage channel.

    Args:
        flow_dirs_filename (str): The filename of the flow directions file.
        sample_sites_filename (str): The filename of the sample sites file.
        drainage_area_threshold (float): The area threshold above which a pixel is classified as a channel.
        plot (bool): Whether to plot the snapped sample sites.
        save (bool): Whether to save the snapped sample sites to a file.
        nudges (Dict[str, np.ndarray]): Dictionary of "nudges" for each sample site. The keys are the sample codes and the values are 2D vectors of dx and dy.
    """
    # Load in real samples
    noisy_samples = pd.read_csv(sample_sites_filename, sep=" ")
    # Check that noisy samples has the correct columns
    if (
        "x_coordinate" not in noisy_samples.columns
        or "y_coordinate" not in noisy_samples.columns
        or "Sample.Code" not in noisy_samples.columns
    ):
        raise ValueError(
            "Noisy samples must have 'x_coordinate', 'y_coordinate' and 'Sample.Code' columns."
        )

    # The D8 accumulator
    accum = D8Accumulator(flow_dirs_filename)
    # Calculate upstream area for each cell
    area = accum.accumulate()
    # Get the x and y coordinates of the channels + combine into an array
    chan_rows, chan_cols = np.where(area > drainage_area_threshold)
    chan_x, chan_y = accum.indices_to_coords(chan_rows, chan_cols)
    chan_coords = np.column_stack((chan_x, chan_y))

    # Initiate an empty dictionary of "nudges", where each sample site is a key pointing to 2D vector of 0s
    all_nudges = {code: np.zeros(2) for code in noisy_samples["Sample.Code"]}
    # Update the nudges dictionary with the nudges argument
    for code, nudge in nudges.items():
        if code not in all_nudges:
            raise ValueError(f"Provided sample code {code} not found in sample sites!")
        # Check that nudge is a 2D vector
        if len(nudge) != 2:
            raise ValueError(f"Nudge for sample code {code} must be a 2D vector.")
        all_nudges[code] = nudge

    initial = np.column_stack((noisy_samples["x_coordinate"], noisy_samples["y_coordinate"]))
    # Nudge the sample according to the entry in "nudges"
    nudged = initial + np.array([all_nudges[code] for code in noisy_samples["Sample.Code"]])
    snapped = np.zeros((noisy_samples.shape[1], 2))
    # Loop through each sample site finding the nearest channel
    for i in range(noisy_samples.shape[1]):
        code = noisy_samples["Sample.Code"][i]
        sample = [noisy_samples["x_coordinate"][i], noisy_samples["y_coordinate"][i]]
        # Nudge the sample according to the entry in "nudges"
        sample = sample + all_nudges[code]
        distances = np.sqrt(np.sum((chan_coords - sample) ** 2, axis=1))
        nearest = chan_coords[np.argmin(distances), :]
        snapped[i] = nearest

    # Plot the network and the noisy, nudged and snapped samples
    if plot:
        plt.figure(figsize=(15, 10))
        plt.title("Snapped Sample Sites")
        plt.scatter(chan_coords[:, 0], chan_coords[:, 1], c="blue", label="Channel")

        # Add a grey line between the noisy and nudged samples
        for i in range(noisy_samples.shape[1]):
            plt.plot(
                [noisy_samples["x_coordinate"][i], nudged[i, 0]],
                [noisy_samples["y_coordinate"][i], nudged[i, 1]],
                c="grey",
                lw=1,
            )
        # Add the nudged samples to the plot
        plt.scatter(nudged[:, 0], nudged[:, 1], c="purple", label="Nudged Sample", marker="x")
        # Add the noisy samples to the plot
        plt.scatter(
            noisy_samples["x_coordinate"],
            noisy_samples["y_coordinate"],
            c="red",
            label="Noisy Sample",
            marker="x",
        )
        # Add a black line between the nudged and snapped samples
        for i in range(noisy_samples.shape[1]):
            plt.plot(
                [nudged[i, 0], snapped[i, 0]],
                [nudged[i, 1], snapped[i, 1]],
                c="black",
                lw=1,
            )
        # Add the snapped samples to the plot
        plt.scatter(snapped[:, 0], snapped[:, 1], c="green", label="Snapped Sample", marker="x")
        plt.legend()
        # Add the sample codes to the plot for each noisy sample
        for i in range(noisy_samples.shape[1]):
            plt.text(
                noisy_samples["x_coordinate"][i],
                noisy_samples["y_coordinate"][i],
                noisy_samples["Sample.Code"][i],
                fontsize=8,
            )
        plt.axis("equal")

    if save:
        # Replace the x and y coordinates of the noisy samples with the snapped ones
        noisy_samples["x_coordinate"] = snapped[:, 0]
        noisy_samples["y_coordinate"] = snapped[:, 1]
        # Save the snapped samples to a file with a suffix "snapped" before the file extension
        outfile = sample_sites_filename.replace(".dat", "_snapped.dat")
        print(f"Saving snapped samples to {outfile}")
        noisy_samples.to_csv(outfile, sep=" ", index=False)


class D8Accumulator:
    """Class to accumulate flow on a D8 flow grid. This class can be used to calculate drainage area and discharge,
    and to accumulate any other tracer across a drainage network. The class assumes that all boundary
    nodes are sinks (i.e., no flow leaves the grid). This class can be used with any geospatial file that GDAL can read.

    Parameters
    ----------
    filename : str
        Path to the D8 flow grid geospatial file (e.g., .tif, .asc, etc.)
        This can be to any file that GDAL can read. Expects a single band raster (ignores other bands).
        This raster should be a 2D array of D8 flow directions according to ESRI convention:

            Sink [no flow]= 0
            Right = 1
            Lower right = 2
            Bottom = 4
            Lower left = 8
            Left = 16
            Upper left = 32
            Top = 64
            Upper right = 128

    Attributes
    ----------
    receivers : np.ndarray
        Array of receiver nodes (i.e., the ID of the node that receives the flow from the i'th node)
    order : np.ndarray
        Array of nodes in order of upstream to downstream (breadth-first)
    baselevel_nodes : np.ndarray
        Array of baselevel nodes (i.e., nodes that do not donate flow to any other nodes)
    arr : np.ndarray
        Array of D8 flow directions
    ds : gdal.Dataset
        GDAL Dataset object of the D8 flow grid. If the array is manually set, this will be None
    extent : List[float]
        Extent of the array in the accumulator as [xmin, xmax, ymin, ymax]. Can be used for plotting.

    Methods
    -------
    accumulate(weights : np.ndarray = None)
        Accumulate flow on the grid using the D8 flow directions
    """

    def __init__(self, filename: str):
        """
        Parameters
        ----------
        filename : str
            Path to the D8 flow grid
        """
        # Check that filename is a string
        if not isinstance(filename, str):
            raise TypeError("Filename must be a string")
        self._arr, self._ds = read_geo_file(filename)
        self._arr = self._arr.astype(int)
        self._receivers = cf.d8_to_receivers(self.arr)
        self._baselevel_nodes = np.where(self.receivers == np.arange(len(self.receivers)))[0]
        self._order = cf.build_ordered_list_iterative(self.receivers, self.baselevel_nodes)

    def accumulate(self, weights: np.ndarray = None) -> np.ndarray:
        """Accumulate flow on the grid using the D8 flow directions

        Parameters
        ----------
        weights : np.ndarray [ndim = 2], optional
            Array of weights for each node, defaults to giving each node a weight of 1, resulting in a map of the number of upstream nodes.
            If the area of each node is known, this can be used to calculate drainage area. If run-off at each node is known,
            this can be used to calculate discharge.

        Returns
        -------
        np.ndarray [ndim = 2]
            Array of accumulated weights (or number of upstream nodes if no weights are passed)
        """
        if weights is None:
            # If no weights are passed, assume all nodes have equal weight of 1.
            # Output is array of # upstream nodes
            weights = np.ones(len(self.receivers))
        else:
            if weights.shape != self.arr.shape:
                raise ValueError("Weights must be have same shape as D8 array")
            weights = weights.flatten()

        return cf.accumulate_flow(self.receivers, self.order, weights=weights).reshape(
            self._arr.shape
        )

    @property
    def receivers(self) -> np.ndarray:
        """Array of receiver nodes (i.e., the ID of the node that receives the flow from the i'th node)"""
        return np.asarray(self._receivers)

    @property
    def baselevel_nodes(self) -> np.ndarray:
        """Array of baselevel nodes (i.e., nodes that do not donate flow to any other nodes)"""
        return self._baselevel_nodes

    @property
    def order(self) -> np.ndarray:
        """Array of nodes in order of upstream to downstream"""
        return np.asarray(self._order)

    @property
    def arr(self):
        """Array of D8 flow directions"""
        return self._arr

    @property
    def ds(self):
        """GDAL Dataset object of the D8 flow grid"""
        return self._ds

    @property
    def extent(self) -> List[float]:
        """
        Get the extent of the array in the accumulator. Can be used for plotting.
        """
        trsfm = self.ds.GetGeoTransform()
        minx = trsfm[0]
        maxy = trsfm[3]
        maxx = minx + trsfm[1] * self.arr.shape[1]
        miny = maxy + trsfm[5] * self.arr.shape[0]
        return [minx, maxx, miny, maxy]

    def indices_to_coords(self, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray]:
        """
        Convert column and row indices to x and y coordinates in the geospatial grid

        Args:
            rows (np.ndarray): Array of row indices
            cols (np.ndarray): Array of column indices

        Returns:
            Tuple[np.ndarray]: Tuple of x and y coordinates
        """

        trsfm = self.ds.GetGeoTransform()
        x = trsfm[0] + cols * trsfm[1]
        y = trsfm[3] + rows * trsfm[5]
        return x, y
