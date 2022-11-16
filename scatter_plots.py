import chart_studio.plotly as py
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

rawData = pd.read_excel("./DraftCityFile.xlsx").sort_values(by = ["destruction"], ascending=False).dropna()
rawData.columns

rawData["pct_old"] = rawData["old_residential"]/rawData["total_residential"]
rawData["inv_dest"] = 1-rawData["destruction"]

xvar = "pct_old"
yvar = "overnights_per_capita_2019"
xtitle = "Percent of Pre-War Housing Surviving"
ytitle = "Average Length of Stay in Days"
"Percent of Pre-War Housing Surviving"
"Average Length of Stay in Days"
"Percent of Housing Stock Not Destroyed"

fig = scatterplot(rawData, xvar, yvar, xtitle, ytitle)
fig.show()

def scatterplot(data, xVar, yVar, xtitle, ytitle):
    xdata = data[xvar]
    ydata = data[yvar]
    size = data["population_2011"]
    size = (size-size.min())/(size.max()-size.min())

    err_size_regr = LinearRegression()
    err_size_res = err_size_regr.fit(np.array(xdata).reshape(-1,1), np.array(ydata))
    err_fit = err_size_regr.predict(np.array(xdata).reshape(-1,1))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    # fig.add_trace(go.Scatter(x=xdata, y=ydata,
    #                     mode='markers',
    #                     name='markers'))

    # fig = go.Figure(data=[go.Scatter(
    #     x=xdata, y=ydata,
    #     mode='markers',
    #     marker_size=size,
    #     text=rawData.columns)
    # ])

    fig.add_trace(
        go.Scatter(
        x=xdata, y=ydata,
        mode='markers',
        marker=dict(
            size=size,
            color = 'rgba(19, 85, 99, 0.8)',
            sizemode='area',
            sizeref=2.*max(size)/(50.**2),
            sizemin=4
        ),
        text = rawData['city'],
        hoverinfo = 'text'),
        )

    # fig.update_traces(hovertemplate=None)
    # fig.update_layout(hovermode=rawData.columns)

    # fig.update_yaxes(rangemode="toone")
    fig.update_layout(
        title="Reported Post War Destruction vs Contemporary Tourist Stays",
        title_x=0.5,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color='rgba(0, 0, 0, 0.8)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_gridcolor='rgba(0,0,0,0.1)',
        yaxis_gridcolor='rgba(0,0,0,0.1)',
        autosize=False, 
        width=1400, 
        height=800
    )

    fig.add_trace(
        go.Scatter(x=xdata, y=err_fit, mode = "lines",name="Error fit", marker_color = 'rgba(122, 34, 15, 0.8)'), 
        secondary_y=False)


    fig.update_layout(showlegend=False)

    return fig
        












