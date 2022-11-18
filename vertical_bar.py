import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

rawData1 = pd.read_excel("./DraftCityFile.xlsx").sort_values(by = ["destruction"], ascending=True).tail(10)
rawData2 = pd.read_excel("./DraftCityFile.xlsx").sort_values(by = ["destruction"], ascending=True).sort_values("population_2011").tail(10)
rawDataMob1 = pd.read_excel("./DraftCityFile.xlsx").sort_values(by = ["destruction"], ascending=True).tail(5)
rawDataMob2 = pd.read_excel("./DraftCityFile.xlsx").sort_values(by = ["destruction"], ascending=True).sort_values("population_2011").tail(5)

figTop10 = destructionBarChart(rawData1)
figTop10.update_layout(
        title={
        'text': "<b>Most Destroyed German Cities</b>",
        'x':0.5})
# figTop10.show()

figPop10 = destructionBarChart(rawData2)
figPop10.update_layout(
        title={
        'text': "<b>Destruction in Large German Cities</b>",
        'x':0.5})
# figPop10.show()

fig3, fig4 = mobileDestructionBarChart(rawDataMob1)
fig3.update_layout(
        title={
        'text': "<b>Wartime Destruction %</b>",
        'x':0.5,
        'font': dict(
         size=20,
        )
        })
fig4.update_layout(
        title={
        'text': "<b>Modern Construction %</b>",
        'font': dict(
         size=20,
        ),
        'x':0.5})
fig3.show()
fig4.show()
fig5, fig6 = mobileDestructionBarChart(rawDataMob2)
fig5.update_layout(
    title={
        'text': "<b>Wartime Destruction %</b>",
        'x':0.5,
        'font': dict(
            size=20,
        )
    })
fig6.update_layout(
        title={
        'text': "<b>Modern Construction %</b>",
        'x':0.5,
        'font': dict(
         size=20,
        )
        })
fig5.show()
fig4.show()

py.plot(figTop10, filename = 'destBar1', auto_open=True)
py.plot(figPop10, filename = 'destBar2', auto_open=True)    

py.plot(fig3, filename = 'destMobBar1', auto_open=False)
py.plot(fig4, filename = 'destMobBar2', auto_open=False)
py.plot(fig5, filename = 'destMobBar3', auto_open=False)
py.plot(fig6, filename = 'destMobBar4', auto_open=False)

def mobileDestructionBarChart(rawData):
    df1_labels = ['Destroyed', 'Remaining']
    df2_labels = ['Pre-War', 'Post-War']

    colors = ['rgba(122, 34, 15, 0.8)', 'rgba(19, 85, 99, 0.8)']

    x1 = rawData[["destruction"]]
    x1["survival"] = 1-x1["destruction"]
    x1 = x1*100
    x1data = x1.to_numpy()

    x2 = pd.DataFrame()
    x2["old_residential"] = rawData["old_residential"].div(rawData["total_residential"].values)
    x2["new_residential"] = 1-x2["old_residential"]
    x2 = x2*100
    x2data = x2.to_numpy()

    ydata = rawData["city"].to_numpy()

    df1Trace1 = go.Bar(
                name=df1_labels[0],
                x=x1['destruction'].to_numpy(), y=ydata,
                orientation='h',
                marker=dict(
                    color=colors[0],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            )

    df1Trace2 = go.Bar(
                name=df1_labels[1],
                x=x1['survival'].to_numpy(), y=ydata,
                orientation='h',
                marker=dict(
                    color=colors[1],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            )

    df2Trace1 = go.Bar(
                name=df2_labels[1],
                x=x2['new_residential'].to_numpy(), y=ydata,
                orientation='h',
                marker=dict(
                    color=colors[0],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            )

    df2Trace2 = go.Bar(
                name=df2_labels[0],
                x=x2['old_residential'].to_numpy(), y=ydata,
                orientation='h',
                marker=dict(
                    color=colors[1],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            )

    fig1 = make_subplots(rows=1, cols=1, specs = [[{}]])
    fig2 = make_subplots(rows=1, cols=1, specs = [[{}]])

    fig1.add_trace(df1Trace1, 1, 1)
    fig1.append_trace(df1Trace2, 1, 1)
    fig2.add_trace(df2Trace1, 1, 1)
    fig2.append_trace(df2Trace2, 1, 1)

    fig1.update_layout(
        margin = {'l': 0, 'r': 0, 't': 50, 'b': 50},
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        barmode='stack',
        title_font_family="Courier New",
        title_font_color="black",
        # title_font_size=25,
        font_family="Courier New",
        font_color="black",
        autosize=True, 
        # width=1200, 
        # height=700
    )

    fig2.update_layout(
        margin = {'l': 0, 'r': 0, 't': 50, 'b': 50},
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            # automargin= True,
            showticklabels=False,
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        barmode='stack',
        title_font_family="Courier New",
        title_font_color="black",
        # title_font_size=25,
        font_family="Courier New",
        font_color="black",
        autosize=True, 
        # width=1200, 
        # height=700
    )

    for i in fig1['layout']['annotations']:
        i['font'] = dict(size=15)
    
    for i in fig2['layout']['annotations']:
        i['font'] = dict(size=15)

    fig1.update_annotations(font=dict(family='Courier New', size=15,
                                        color='rgb(67, 67, 67)'))
    
    fig2.update_annotations(font=dict(family='Courier New', size=15,
                                    color='rgb(67, 67, 67)'))

    for yd, xd in zip(ydata, x1data):
        # labeling the y-axis
        # fig1.add_annotation(dict(xref='paper', yref='y',
        #                         x=-0.01, y=yd,
        #                         xanchor='right',
        #                         text='<b>'+str(yd)+'</b>',
        #                         font=dict(family='Courier New', size=15,
        #                                 color='rgb(67, 67, 67)'),
        #                         showarrow=False, align='right'))

        # labeling the first percentage of each bar (x_axis)
        fig1.add_annotation(dict(xref='x', yref='y',
                                x=xd[0] / 2, y=yd,
                                text='<b>'+str(yd)+':</b>'+str(int(xd[0])) + '%',
                                font=dict(family='Courier New', size=15,
                                        color='rgb(248, 248, 255)'),
                                showarrow=False))
    
    for yd, xd in zip(ydata, x2data):
        # labeling the y-axis
        # fig2.add_annotation(dict(xref='paper', yref='y',
        #                         x=-0.01, y=yd,
        #                         xanchor='right',
        #                         text='<b>'+str(yd)+'</b>',
        #                         font=dict(family='Courier New', size=15,
        #                                 color='rgb(67, 67, 67)'),
        #                         showarrow=False, align='right'))

        # labeling the first percentage of each bar (x_axis)
        fig2.add_annotation(dict(xref='x', yref='y',
                                x=xd[1] / 2, y=yd,
                                text='<b>'+str(yd)+':</b>'+str(int(xd[1])) + '%',
                                font=dict(family='Courier New', size=15,
                                        color='rgb(248, 248, 255)'),
                                showarrow=False))
        
    return fig1, fig2


def destructionBarChart(rawData):

    df1_labels = ['Destroyed', 'Remaining']
    df2_labels = ['Pre-War', 'Post-War']

    colors = ['rgba(122, 34, 15, 0.8)', 'rgba(19, 85, 99, 0.8)']

    x1 = rawData[["destruction"]]
    x1["survival"] = 1-x1["destruction"]
    x1 = x1*100
    x1data = x1.to_numpy()

    x2 = pd.DataFrame()
    x2["old_residential"] = rawData["old_residential"].div(rawData["total_residential"].values)
    x2["new_residential"] = 1-x2["old_residential"]
    x2 = x2*100
    x2data = x2.to_numpy()

    ydata = rawData["city"].to_numpy()

    df1Trace1 = go.Bar(
                name=df1_labels[0],
                x=x1['destruction'].to_numpy(), y=ydata,
                orientation='h',
                offsetgroup=1,
                marker=dict(
                    color=colors[0],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            )

    df1Trace2 = go.Bar(
                name=df1_labels[1],
                x=x1['survival'].to_numpy(), y=ydata,
                orientation='h',
                offsetgroup=1,
                marker=dict(
                    color=colors[1],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            )

    df2Trace1 = go.Bar(
                name=df2_labels[1],
                x=x2['new_residential'].to_numpy(), y=ydata,
                orientation='h',
                offsetgroup=2,
                marker=dict(
                    color=colors[0],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            )

    df2Trace2 = go.Bar(
                name=df2_labels[0],
                x=x2['old_residential'].to_numpy(), y=ydata,
                orientation='h',
                offsetgroup=2,
                marker=dict(
                    color=colors[1],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            )


    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,  
            subplot_titles=("<b>Wartime Destruction %</b>", "<b>Modern Construction %</b>"), 
            specs = [[{}, {}]],
                            horizontal_spacing = 0.05)

    fig.add_trace(df1Trace1, 1, 1)
    fig.append_trace(df1Trace2, 1, 1)
    fig.add_trace(df2Trace1, 1, 2)
    fig.append_trace(df2Trace2, 1, 2)

    fig.update_layout(
        margin = {'l': 200, 'r': 50, 't': 100, 'b': 50},
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            automargin= True,
            showticklabels=False,
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        barmode='stack',
        title_font_family="Courier New",
        title_font_color="black",
        title_font_size=25,
        font_family="Courier New",
        font_color="black",
        # autosize=False, 
        # width=1200, 
        # height=700
    )

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=20)

    fig.update_annotations(font=dict(family='Courier New', size=18,
                                        color='rgb(67, 67, 67)'))

    annotations = []

    row = 0
    for yd, xd in zip(ydata, x1data):
        # labeling the y-axis
        fig.add_annotation(dict(xref='paper', yref='y',
                                x=-0.01, y=yd,
                                xanchor='right',
                                text='<b>'+str(yd)+'</b>',
                                font=dict(family='Courier New', size=18,
                                        color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        if xd[0] > 9:
            # labeling the first percentage of each bar (x_axis)
            fig.add_annotation(dict(xref='x', yref='y',
                                    x=xd[0] / 2, y=yd,
                                    text=str(int(xd[0])) + '%',
                                    font=dict(family='Courier New', size=18,
                                            color='rgb(248, 248, 255)'),
                                    showarrow=False))
        space = xd[0]

        if xd[1] > 9:
            # labeling the second percentages for each bar (x_axis)
            fig.add_annotation(dict(xref='x', yref='y',
                                    x=space + xd[1]/2, y=yd,
                                    text=str(int(xd[1])) + '%',
                                    font=dict(family='Courier New', size=18,
                                                color='rgb(248, 248, 255)'),
                                    showarrow=False))
        

        if x2data[row][1] > 9:
            # labeling the third percentage of each bar (x_axis)
            fig.add_annotation(dict(xref='x2', yref='y',
                                    x=x2data[row][1] / 2, y=yd,
                                    text=str(int(x2data[row][1])) + '%',
                                    font=dict(family='Courier New', size=18,
                                            color='rgb(248, 248, 255)'),
                                    showarrow=False))
        space =  x2data[row][1]

        if x2data[row][0] > 9:
            # labeling the rest of percentages for each bar (x_axis)
            fig.add_annotation(dict(xref='x2', yref='y',
                                    x=space + x2data[row][0]/2, y=yd,
                                    text=str(int(x2data[row][0])) + '%',
                                    font=dict(family='Courier New', size=18,
                                                color='rgb(248, 248, 255)'),
                                    showarrow=False))
        space += x2data[row][0]
        row += 1
    return fig

