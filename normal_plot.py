import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

rawData = pd.read_excel("./DraftCityFile.xlsx").sort_values(by = ["destruction"], ascending=True)

x1 = rawData[["destruction"]]
x1["survival"] = 1-x1["destruction"]
x1 = x1*100
x1["very_old_residential"] = rawData["very_old_residential"]
x1["old_per_km2"] = rawData["very_old_residential"].div(rawData["area_km2"].values)


group_labels = ['Level of Destruction', 'Old Buildings per Km2']

colors = ['rgba(122, 34, 15, 0.8)', 'rgba(19, 85, 99, 0.8)']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot([x1["destruction"], x1["old_per_km2"]], group_labels, bin_size=3,
                         curve_type='normal', # override default 'kde'
                         colors=colors, show_rug=False)

# Add title
fig.update_layout(title_text='Normal Distribution Plot', title_x=0.5, title_y=1)

fig.update_layout(
    margin = {'l': 50, 'r': 0, 't': 50, 'b': 50},
    xaxis=dict(
        showgrid=False,
        showline=False,
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        zeroline=False,
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title_font_family="Courier New",
    title_font_color="black",
    title_font_size=25,
    font_family="Courier New",
    font_color="black",
    # autosize=False, 
    # width=1200, 
    # height=700,
    legend=dict(
        traceorder="normal",
        font=dict(
            family="Courier New",
            size=15,
            color="black"
        ),
        orientation="h",
        yanchor="bottom",
        y=.96,
        xanchor="right",
        x=1
    )
)

fig.for_each_xaxis(lambda x: x.update(showgrid=True, showline=True, linewidth=1, linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)', rangemode = "tozero"))
fig.for_each_yaxis(lambda x: x.update(showgrid=True, showline=True, linewidth=1, linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)', rangemode = "tozero"))


py.plot(fig, filename = 'normDist1', auto_open=True)
# fig.show()