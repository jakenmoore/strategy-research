import pandas as pd

# charting libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def plotly_line_chart(
    data: pd.DataFrame, column1: str, column2: str = "", secondary_y: bool = False
) -> go.Figure:
    """
    returns a plotly chart with either on or two columns
    on one or two axes

    :type data: dataframe object
    :param column1: string name of column in dataframe
    :param column2: string name of column in dataframe
    :type secondary_y: boolean object False by default, set to True for two axes

    """
    fig = make_subplots(specs=[[{"secondary_y": secondary_y}]])
    df = data[[column1]]
    fig.add_trace(
        go.Scatter(x=df.index, y=df[column1], mode="lines", name=column1),
        secondary_y=False,
    )
    if column2 is not None:
        df = data[[column1, column2]]

        fig.add_trace(
            go.Scatter(x=df.index, y=df[column2], mode="lines", name=column2),
            secondary_y=secondary_y,
        )

    return fig


def lag_price(
    data, column="price", windows=None
):  # the unit is the resample period i.e. 10ms
    if windows is None:
        windows = [-100, -500, -1000, -3000, -6000]
    for window in windows:
        data["lagged_price_{}0_ms".format(window)] = data[column].shift(window)
    return
