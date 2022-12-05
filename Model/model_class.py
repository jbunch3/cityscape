import chart_studio.plotly as py
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
import plotly.figure_factory as ff
from sklearn.preprocessing import MaxAbsScaler
import plotly.express as px
import statsmodels.api as sm


class VariableSet():
    def __init__(self, xVar: str = "", yVar: str = "", xNames: str = "", yNames: str = "", zNames: str = "", zVar: str = "", xTitle: str = "", yTitle: str = "", gTitle: str = ""):
        self.xVar = xVar
        self.yVar = yVar
        self.zVar = zVar
        self.xNames = xNames
        self.yNames = yNames
        self.zNames = zNames
        self.xTitle = xTitle
        self.yTitle = yTitle
        self.gTitle = gTitle


class Model():
    def __init__(self, layout: dict, data: pd.DataFrame):
        self.layout = layout
        self.data = data

    def olstrace(self, variableSet: VariableSet):
        xdata = self.data[variableSet.xVar]
        ydata = self.data[variableSet.yVar]
        err_size_regr = LinearRegression()
        err_size_regr.fit(np.array(xdata).reshape(-1, 1), np.array(ydata))
        err_fit = err_size_regr.predict(np.array(xdata).reshape(-1, 1))

        return go.Scatter(x=xdata, y=err_fit, mode="lines", name="OLS Fit", marker_color='rgba(122, 34, 15, 0.8)')

    def QuadScatterPlot(self, mobile: bool,  layout: dict, variableSets: List[VariableSet]):
        v1, v2, v3, v4 = variableSets

        if mobile:
            fig = make_subplots(rows=4, cols=1, shared_yaxes=False, shared_xaxes=False,
                                subplot_titles=(v1.gTitle, v2.gTitle,
                                                v3.gTitle, v4.gTitle),
                                specs=[[{}], [{}], [{}], [{}]], horizontal_spacing=0.01, vertical_spacing=0.1)

            fig.add_trace(self.SubScatterPlot(v1), 1, 1)
            fig.add_trace(self.SubScatterPlot(v2), 2, 1)
            fig.add_trace(self.SubScatterPlot(v3), 3, 1)
            fig.add_trace(self.SubScatterPlot(v4), 4, 1)

            fig.add_trace(self.olstrace(v1), 1, 1)
            fig.add_trace(self.olstrace(v2), 2, 1)
            fig.add_trace(self.olstrace(v3), 3, 1)
            fig.add_trace(self.olstrace(v4), 4, 1)

            # Update xaxis properties
            fig.update_xaxes(title_text=v1.xTitle,
                             showgrid=True, row=1, col=1)
            fig.update_xaxes(title_text=v2.xTitle,
                             showgrid=True,  row=2, col=1)
            fig.update_xaxes(title_text=v3.xTitle,
                             showgrid=True, row=3, col=1)
            fig.update_xaxes(title_text=v4.xTitle,
                             showgrid=True,  row=4, col=1)

            # Update yaxis properties
            fig.update_yaxes(title_text=v1.yTitle,
                             showgrid=True, row=1, col=1)
            fig.update_yaxes(title_text=v2.yTitle,
                             showgrid=True, row=2, col=1)
            fig.update_yaxes(title_text=v3.yTitle,
                             showgrid=True, row=3, col=1)
            fig.update_yaxes(title_text=v4.yTitle,
                             showgrid=True, row=4, col=1)
        else:

            fig = make_subplots(rows=2, cols=2, shared_yaxes=True, shared_xaxes=True,
                                subplot_titles=(v1.gTitle, v2.gTitle,
                                                v3.gTitle, v4.gTitle),
                                specs=[[{}, {}], [{}, {}]], horizontal_spacing=0.01, vertical_spacing=0.06)

            fig.add_trace(self.SubScatterPlot(v1), 1, 1)
            fig.add_trace(self.SubScatterPlot(v2), 1, 2)
            fig.add_trace(self.SubScatterPlot(v3), 2, 1)
            fig.add_trace(self.SubScatterPlot(v4), 2, 2)

            fig.add_trace(self.olstrace(v1), 1, 1)
            fig.add_trace(self.olstrace(v2), 1, 2)
            fig.add_trace(self.olstrace(v3), 2, 1)
            fig.add_trace(self.olstrace(v4), 2, 2)

            # Update xaxis properties
            fig.update_xaxes(title_text=v1.xTitle,
                             showgrid=True, row=1, col=1)
            fig.update_xaxes(title_text=v2.xTitle,
                             showgrid=True,  row=1, col=3)
            fig.update_xaxes(title_text=v3.xTitle,
                             showgrid=True, row=2, col=1)
            fig.update_xaxes(title_text=v4.xTitle,
                             showgrid=True,  row=2, col=3)

            # Update yaxis properties
            fig.update_yaxes(title_text=v1.yTitle,
                             showgrid=True, row=1, col=1)
            fig.update_yaxes(title_text=v2.yTitle,
                             showgrid=True, row=1, col=3)
            fig.update_yaxes(title_text=v3.yTitle,
                             showgrid=True, row=2, col=1)
            fig.update_yaxes(title_text=v4.yTitle,
                             showgrid=True, row=2, col=3)

        fig.update_layout({**self.layout, **layout})

        fig.update_traces(
            hovertemplate='City: %{x} <br>Pop: %{y}', selector={'': ''})

        fig.for_each_xaxis(lambda x: x.update(showgrid=True, showline=True,
                           linewidth=1, linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)'))
        fig.for_each_yaxis(lambda x: x.update(showgrid=True, showline=True,
                           linewidth=1, linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)'))

        return fig

    def SubScatterPlot(self, variableSet: VariableSet):
        xdata = self.data[variableSet.xVar]
        ydata = self.data[variableSet.yVar]
        scale = self.data[variableSet.zVar]
        scale = (scale-scale.min())/(scale.max()-scale.min())

        trace = go.Scatter(
            x=xdata, y=ydata,
            mode='markers',
            marker=dict(
                size=scale,
                color='rgba(19, 85, 99, 0.8)',
                sizemode='area',
                sizeref=2.*max(scale)/(50.**2),
                sizemin=4
            ),
            text=self.data[variableSet.zNames],
            hoverinfo='text'
        )

        return trace

    def FancyScatterPlot(self, layout: dict, variableSet: VariableSet):
        xdata = self.data[variableSet.xVar]
        ydata = self.data[variableSet.yVar]
        scale = self.data[variableSet.zVar]
        scale = (scale-scale.min())/(scale.max()-scale.min())

        olsLine = self.olstrace(variableSet)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=xdata, y=ydata,
                mode='markers',
                marker=dict(
                    size=scale,
                    color='rgba(19, 85, 99, 0.8)',
                    sizemode='area',
                    sizeref=2.*max(scale)/(50.**2),
                    sizemin=4
                ),
                text=self.data[variableSet.zNames],
                hoverinfo='text'),
        )

        fig.update_layout({**self.layout, **layout})

        fig.update_layout(
            title=variableSet.gTitle,
            title_x=0.5,
            xaxis_title=variableSet.xTitle,
            yaxis_title=variableSet.yTitle,
        )

        fig.add_trace(olsLine)

        fig.update_layout(showlegend=False)

        return fig

    def SimpleNormalDistOverlay(self, layout: dict, variableSets: List[VariableSet]):
        colors = ['rgba(122, 34, 15, 0.8)', 'rgba(19, 85, 99, 0.8)']
        stdVars = pd.DataFrame([])

        stdVars[[variableSets[0].xVar, variableSets[1].xVar]] = MaxAbsScaler(
        ).fit_transform(self.data[[variableSets[0].xVar, variableSets[1].xVar]])

        # Create distplot with curve_type set to 'normal'
        fig = ff.create_distplot([stdVars[variableSets[0].xVar], stdVars[variableSets[1].xVar]],
                                 [variableSets[0].xTitle,
                                     variableSets[1].xTitle], bin_size=.05,
                                 curve_type='normal',
                                 colors=colors,
                                 show_rug=False)

        # Add title
        fig.update_layout({**self.layout, **layout})
        fig.update_layout(title_text='Normal Distribution Plot',
                          title_x=0.5, title_y=1)

        fig.for_each_xaxis(lambda x: x.update(showgrid=True, showline=True, linewidth=1,
                           linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)', rangemode="tozero"))
        fig.for_each_yaxis(lambda x: x.update(showgrid=True, showline=True, linewidth=1,
                           linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)', rangemode="tozero"))

        return fig

    def CorrelationHeatPlot(self, mobile: bool, layout: dict, variableSets: List[VariableSet]):

        varlist = list(map(lambda var: var.xVar, variableSets))
        corrNames = list(map(lambda var: var.xTitle, variableSets))

        corr = self.data[varlist].dropna().corr(numeric_only=True)

        mask = np.triu(np.ones_like(corr, dtype=bool))

        corrplot = go.Heatmap(
            z=corr.mask(mask),
            x=corrNames,
            y=corrNames,
            colorscale=px.colors.diverging.RdBu,
            zmin=-1,
            zmax=1
        )

        corrplotMob = go.Heatmap(
            z=corr.mask(mask),
            x=corrNames,
            y=corrNames,
            colorscale=px.colors.diverging.RdBu,
            showscale=False,
            zmin=-1,
            zmax=1
        )

        m_layout = {
            'xaxis': dict(
                tickangle=90,
            ),
            'yaxis': dict(
                showticklabels=False,
            ),
        }

        fig = go.Figure(data=[corrplot], layout={**self.layout, **layout})
        if mobile:
            fig = go.Figure(data=[corrplotMob], layout={
                            **self.layout, **layout, **m_layout})

        return fig

    def ScatterMatrix(self,  VariableSets: List[VariableSet], layout: dict, size: int):
        varNames = {}
        for var in VariableSets:
            varNames[var.xVar] = var.xTitle

        data = self.data[list(map(lambda var: var.xVar, VariableSets))]
        data = data.rename(varNames, axis='columns')

        fig = ff.create_scatterplotmatrix(
            df=data,
            diag='histogram')

        fig = fig.update_layout({**self.layout, **layout})
        fig.for_each_xaxis(lambda x: x.update(showgrid=True, showline=True, linewidth=1,
                                              linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)', rangemode="tozero"))
        fig.for_each_yaxis(lambda x: x.update(showgrid=True, showline=True, linewidth=1,
                           linecolor='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.1)', rangemode="tozero"))
        fig.update_layout(title_text='Scatter Matrix', title_x=0.5)
        fig = fig.update_layout(
            {'width': size, 'height': size, 'autosize': True})

        return fig

    def LinearModel(self, yVarSet: VariableSet, XVariableSets: List[VariableSet]):

        varlist = list(map(lambda var: var.xVar, XVariableSets))

        data = self.data.dropna()

        y = data[[yVarSet.yVar]].apply(pd.to_numeric)
        X = data[varlist].apply(pd.to_numeric)

        X = sm.add_constant(X)
        mod = sm.OLS(y, X)
        res = mod.fit(cov_type='HC3', use_t=True)

        return res
