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

rawData["old_per_km2"] = rawData["very_old_residential"].div(rawData["area_km2"].values)
rawData["old_per_capita"] = rawData["very_old_residential"].div(rawData["population_2011"].values)
rawData["overnights_per_capita_2021"] = rawData["overnights_2021"].div(rawData["population_2011"].values)
rawData["instagram_post_count_cap"] = rawData["instagram_post_count"].div(rawData["population_2011"].values)

yvar = "overnights_per_capita_2019"
xvar = "instagram_post_count_cap"
ytitle = "Overnights per Capita"
xtitle = "Instagram Posts per Capita"
"Percent of Pre-War Housing Surviving"
"Average Length of Stay in Days"
"Percent of Housing Stock Not Destroyed"

fig = scatterplot(rawData, xvar, yvar, xtitle, ytitle)
py.plot(fig, filename = 'scatter2', auto_open=True)
# fig.show()

fig = make_subplots(rows=2, cols=2, shared_yaxes=True, shared_xaxes=True,
        subplot_titles=("<b>Old Buildings & Instagram</b>", "<b>Wartime Destruction & Instagram</b>", "<b>Old Buildings & Overnight Stays</b>", "<b>Wartime Destruction & Overnight Stays</b>"), 
        specs = [[{}, {}],[{}, {}]], horizontal_spacing = 0.01, vertical_spacing = 0.06)

fig.add_trace(scattertrace(rawData, "old_per_capita", "instagram_post_count_cap"), 1, 1)
fig.add_trace(scattertrace(rawData, "destruction", "instagram_post_count_cap"), 1, 2)
fig.add_trace(scattertrace(rawData, "old_per_capita", "overnights_per_capita_2019"), 2, 1)
fig.add_trace(scattertrace(rawData, "destruction", "overnights_per_capita_2019"), 2, 2)

fig.add_trace(olstrace(rawData, "old_per_capita", "instagram_post_count_cap"), 1, 1)
fig.add_trace(olstrace(rawData, "destruction", "instagram_post_count_cap"), 1, 2)
fig.add_trace(olstrace(rawData, "old_per_capita", "overnights_per_capita_2019"), 2, 1)
fig.add_trace(olstrace(rawData, "destruction", "overnights_per_capita_2019"), 2, 2)

# Update xaxis properties
# fig.update_xaxes(title_text="xaxis 1 title", showgrid=True, row=1, col=1)
# fig.update_xaxes(title_text="xaxis 2 title", showgrid=True, row=1, col=2)
fig.update_xaxes(title_text="Old Buildings per Capita", showgrid=True, row=2, col=1)
fig.update_xaxes(title_text="Wartime Destruction (%)", showgrid=True,  row=2, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Instagram Posts per Capita", showgrid=True, row=1, col=1)
# fig.update_yaxes(title_text="yaxis 2 title", showgrid=True, row=1, col=2)
fig.update_yaxes(title_text="Overnights per Capita", showgrid=True, row=2, col=1)
# fig.update_yaxes(title_text="yaxis 4 title", showgrid=True, row=2, col=2)

fig.update_layout(
    showlegend=False,
    font=dict(
        family="Courier New, monospace",
        size=14,
        color='rgba(0, 0, 0, 0.8)'
    ),
    hoverlabel=dict(
        bgcolor="rgba(0,0,0,0.1)",
        font_size=18,
        font_family="Courier New"
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    # autosize=False, 
    # width=1200, 
    # height=1200
)

fig.update_traces(hovertemplate='City: %{x} <br>Pop: %{y}', selector={'':''})

fig.for_each_xaxis(lambda x: x.update(showgrid=True, showline=True, linewidth=1, linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)'))
fig.for_each_yaxis(lambda x: x.update(showgrid=True, showline=True, linewidth=1, linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)'))

# py.plot(fig, filename = 'scatter1', auto_open=True)
fig.show()

fig = make_subplots(rows=4, cols=1, shared_yaxes=False, shared_xaxes=False,
        subplot_titles=("<b>Buildings & Instagram</b>", "<b>Destruction & Instagram</b>", "<b>Buildings & Overnights</b>", "<b>Destruction & Overnights</b>"), 
        specs = [[{}],[{}],[{}],[{}]], horizontal_spacing = 0.01, vertical_spacing = 0.1)

fig.add_trace(scattertrace(rawData, "old_per_capita", "instagram_post_count_cap"), 1, 1)
fig.add_trace(scattertrace(rawData, "destruction", "instagram_post_count_cap"), 2, 1)
fig.add_trace(scattertrace(rawData, "old_per_capita", "overnights_per_capita_2019"), 3, 1)
fig.add_trace(scattertrace(rawData, "destruction", "overnights_per_capita_2019"), 4, 1)

fig.add_trace(olstrace(rawData, "old_per_capita", "instagram_post_count_cap"), 1, 1)
fig.add_trace(olstrace(rawData, "destruction", "instagram_post_count_cap"), 2, 1)
fig.add_trace(olstrace(rawData, "old_per_capita", "overnights_per_capita_2019"), 3, 1)
fig.add_trace(olstrace(rawData, "destruction", "overnights_per_capita_2019"), 4, 1)

# Update xaxis properties
fig.update_xaxes(title_text="Old Buildings p. C.", showgrid=True, row=1, col=1)
fig.update_xaxes(title_text="Destruction (%)", showgrid=True,  row=2, col=1)
fig.update_xaxes(title_text="Old Buildings  p. C.", showgrid=True, row=3, col=1)
fig.update_xaxes(title_text="Destruction (%)", showgrid=True,  row=4, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Instagram Posts p. C.", showgrid=True, row=1, col=1)
fig.update_yaxes(title_text="Instagram Posts p. C.", showgrid=True, row=2, col=1)
fig.update_yaxes(title_text="Overnights p. C.", showgrid=True, row=3, col=1)
fig.update_yaxes(title_text="Overnights p. C.", showgrid=True, row=4, col=1)

fig.update_layout(
    showlegend=False,
    margin = {'l': 0, 'r': 0, 't': 50, 'b': 50},
    font=dict(
        family="Courier New, monospace",
        size=12,
        color='rgba(0, 0, 0, 0.8)'
    ),
    hoverlabel=dict(
        bgcolor="rgba(0,0,0,0.1)",
        font_size=18,
        font_family="Courier New"
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    autosize=True, 
    # width=1200, 
    # height=1200
)

fig.update_traces(hovertemplate='City: %{x} <br>Pop: %{y}', selector={'':''})

fig.for_each_xaxis(lambda x: x.update(showgrid=True, showline=True, linewidth=1, linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)'))
fig.for_each_yaxis(lambda x: x.update(showgrid=True, showline=True, linewidth=1, linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)'))

py.plot(fig, filename = 'scatterMob1', auto_open=True)
fig.show()

def scattertrace(data, xVar, yVar):
    xdata = data[xVar]
    ydata = data[yVar]
    size = data["population_2011"]
    size = (size-size.min())/(size.max()-size.min())

    trace = go.Scatter(
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
        hoverinfo = 'text'
    )

    return trace

def olstrace(data, xVar, yVar):
    xdata = data[xVar]
    ydata = data[yVar]
    err_size_regr = LinearRegression()
    err_size_regr.fit(np.array(xdata).reshape(-1,1), np.array(ydata))
    err_fit = err_size_regr.predict(np.array(xdata).reshape(-1,1))

    return go.Scatter(x=xdata, y=err_fit, mode = "lines",name="Error fit", marker_color = 'rgba(122, 34, 15, 0.8)')






def scatterplot(data, xVar, yVar, xtitle, ytitle):
    xdata = data[xVar]
    ydata = data[yVar]
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
        margin = {'l': 50, 'r': 0, 't': 50, 'b': 50},
        title="Instagram vs Overnights",
        title_x=0.5,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color='rgba(0, 0, 0, 0.8)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_gridcolor='rgba(0,0,0,0.1)',
        yaxis_gridcolor='rgba(0,0,0,0.1)',
        # autosize=False, 
        # width=1400, 
        # height=800
    )

    fig.add_trace(
        go.Scatter(x=xdata, y=err_fit, mode = "lines",name="Error fit", marker_color = 'rgba(122, 34, 15, 0.8)'), 
        secondary_y=False)


    fig.update_layout(showlegend=False)

    return fig
        












