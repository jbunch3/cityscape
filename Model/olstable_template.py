
from string import Template
from Model.model_class import VariableSet
from typing import List


def LatexOLSTableOut(title: str, variableSets: List[VariableSet], results):
    tableOut = Template(r'''
                \documentclass{article} 
                \usepackage[utf8]{inputenc} 
                \\
                \title{Test} 
                \author{Justin Bunch} 
                \date{November 2022} 
                \\
                \begin{document} 
                \newpage
                \\
                \begin{center} 
                \Large\textbf{$title} 
                \\ 
                \begin{tabular}{lc} 
                \hline 
                \textbf{Model:}  & OLS \\ 
                \textbf{Method:} & Least Squares \\  
                \textbf{No. Observations:} & $nObs \\  
                \textbf{R-squared:} & $r2   \\ 
                \textbf{Adj. R-squared:} & $adjR2 \\ 
                \textbf{F-statistic:} & $fstat \\ 
                \textbf{Prob (F-statistic):} &  $pFstat \\  
                \textbf{Covariance Type:} & Heteroscedasticity robust (HC3) \\  
                \textbf{ } & \\ 
                \end{tabular} 
                \begin{tabular}{lcccc} 
                & \textbf{Coef} & \textbf{Std. Err} & \textbf{t} & \textbf{P > |t|}  \\ 
                \hline 
                \textbf{Const} & $cCoef & $cStdErr & $cTval & $cPT \\ 
                \textbf{$var1} & $v1Coef & $v1StdErr & $v1Tval & $v1cPT \\ 
                \textbf{$var2} & $v2Coef & $v2StdErr & $v2Tval & $v2cPT \\ 
                \textbf{$var3} & $v3Coef & $v3StdErr & $v3Tval & $v3cPT \\ 
                \textbf{$var4} & $v4Coef & $v4StdErr & $v4Tval & $v4cPT \\ 
                \textbf{$var5} & $v5Coef & $v5StdErr & $v5Tval & $v5cPT \\ 
                \textbf{$var6} & $v6Coef & $v6StdErr & $v6Tval & $v6cPT \\ 
                \textbf{$var7} & $v7Coef & $v7StdErr & $v7Tval & $v7cPT \\ 
                \textbf{$var8} & $v8Coef & $v8StdErr & $v8Tval & $v8cPT \\ 
                \textbf{$var9} & $v9Coef & $v9StdErr & $v9Tval & $v9cPT \\ 
                \hline 
                \end{tabular} 
                \end{center} 

                \end{document}
                ''')

    coefs = results.params.map(lambda p: "{:.3f}".format(p))
    stdErrs = results.HC3_se.map(lambda p: "{:.3f}".format(p))
    tVals = results.tvalues.map(lambda p: "{:.3f}".format(p))
    pTVals = results.pvalues.map(lambda p: "{:.3f}".format(p))

    tableOut = tableOut.substitute(
        title=title, nObs="{:.3f}".format(results.nobs), r2="{:.3f}".format(results.rsquared), adjR2="{:.3f}".format(results.rsquared_adj), fstat="{:.3f}".format(results.fvalue), pFstat="{:.3f}".format(results.f_pvalue),
        var1=variableSets[0].xTitle, var2=variableSets[1].xTitle, var3=variableSets[
            2].xTitle, var4=variableSets[3].xTitle, var5=variableSets[4].xTitle, var6=variableSets[5].xTitle,  var7=variableSets[6].xTitle, var8=variableSets[7].xTitle, var9=variableSets[8].xTitle,
        cCoef=coefs[0], v1Coef=coefs[1], v2Coef=coefs[2], v3Coef=coefs[3], v4Coef=coefs[
            4], v5Coef=coefs[5], v6Coef=coefs[6], v7Coef=coefs[7], v8Coef=coefs[8], v9Coef=coefs[9],
        cStdErr=stdErrs[0], v1StdErr=stdErrs[1], v2StdErr=stdErrs[2], v3StdErr=stdErrs[
            3], v4StdErr=stdErrs[4], v5StdErr=stdErrs[5], v6StdErr=stdErrs[6], v7StdErr=stdErrs[7], v8StdErr=stdErrs[8], v9StdErr=stdErrs[9],
        cTval=tVals[0], v1Tval=tVals[1], v2Tval=tVals[2], v3Tval=tVals[3], v4Tval=tVals[
            4], v5Tval=tVals[5], v6Tval=tVals[6], v7Tval=tVals[7], v8Tval=tVals[8], v9Tval=tVals[9],
        cPT=pTVals[0], v1cPT=pTVals[1], v2cPT=pTVals[2], v3cPT=pTVals[3], v4cPT=pTVals[4], v5cPT=pTVals[5], v6cPT=pTVals[6], v7cPT=pTVals[7], v8cPT=pTVals[8], v9cPT=pTVals[9])

    return tableOut
