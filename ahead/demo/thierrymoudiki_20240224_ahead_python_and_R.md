# 0 - Install and import packages


```python
!pip install ahead
```


```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

from time import time
from ahead import Ridge2Regressor # may take some time to run, ONLY the 1st time it's run
from statsmodels.tsa.base.datetools import dates_from_str
from time import time
```

# 1 - Import data


```python
# some example data
mdata = sm.datasets.macrodata.load_pandas().data

# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)

quarterly = dates["year"] + "Q" + dates["quarter"]

quarterly = dates_from_str(quarterly)

mdata = mdata[['realgovt', 'tbilrate', 'cpi', 'realgdp']]

mdata.index = pd.DatetimeIndex(quarterly)

df = np.log(mdata).diff().dropna()

display(df)
```



  <div id="df-f2e345a0-7555-4e1a-ba7c-9f31c295667f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>realgovt</th>
      <th>tbilrate</th>
      <th>cpi</th>
      <th>realgdp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1959-06-30</th>
      <td>0.023664</td>
      <td>0.088193</td>
      <td>0.005849</td>
      <td>0.024942</td>
    </tr>
    <tr>
      <th>1959-09-30</th>
      <td>0.020481</td>
      <td>0.215321</td>
      <td>0.006838</td>
      <td>-0.001193</td>
    </tr>
    <tr>
      <th>1959-12-31</th>
      <td>-0.014781</td>
      <td>0.125317</td>
      <td>0.000681</td>
      <td>0.003495</td>
    </tr>
    <tr>
      <th>1960-03-31</th>
      <td>-0.046197</td>
      <td>-0.212805</td>
      <td>0.005772</td>
      <td>0.022190</td>
    </tr>
    <tr>
      <th>1960-06-30</th>
      <td>-0.003900</td>
      <td>-0.266946</td>
      <td>0.000338</td>
      <td>-0.004685</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2008-09-30</th>
      <td>0.031005</td>
      <td>-0.396881</td>
      <td>-0.007904</td>
      <td>-0.006781</td>
    </tr>
    <tr>
      <th>2008-12-31</th>
      <td>0.015732</td>
      <td>-2.277267</td>
      <td>-0.021979</td>
      <td>-0.013805</td>
    </tr>
    <tr>
      <th>2009-03-31</th>
      <td>-0.010967</td>
      <td>0.606136</td>
      <td>0.002340</td>
      <td>-0.016612</td>
    </tr>
    <tr>
      <th>2009-06-30</th>
      <td>0.026975</td>
      <td>-0.200671</td>
      <td>0.008419</td>
      <td>-0.001851</td>
    </tr>
    <tr>
      <th>2009-09-30</th>
      <td>0.019888</td>
      <td>-0.405465</td>
      <td>0.008894</td>
      <td>0.006862</td>
    </tr>
  </tbody>
</table>
<p>202 rows × 4 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f2e345a0-7555-4e1a-ba7c-9f31c295667f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f2e345a0-7555-4e1a-ba7c-9f31c295667f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f2e345a0-7555-4e1a-ba7c-9f31c295667f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-71b61756-860a-4500-a599-f25d8e609908">
  <button class="colab-df-quickchart" onclick="quickchart('df-71b61756-860a-4500-a599-f25d8e609908')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-71b61756-860a-4500-a599-f25d8e609908 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




```python
regr = Ridge2Regressor(h = 20, date_formatting = "original",
                       type_pi="blockbootstrap", # type of simulations for predictive inference
                       B=50, # number of simulations
                       seed=1)

start = time()
regr.forecast(df)
print(f"Elapsed: {time() - start} s")
```

      |======================================================================| 100%
    Elapsed: 0.49063634872436523 s



```python
regr2 = Ridge2Regressor(h = 20, date_formatting = "original",
                       type_pi="gaussian", # Gaussian prediction intervals
                       seed=1)

start = time()
regr2.forecast(df)
print(f"Elapsed: {time() - start} s")
```

    Elapsed: 0.038446664810180664 s



```python
regr.plot("realgovt")
```


    
![png](thierrymoudiki_20240224_ahead_python_and_R_files/thierrymoudiki_20240224_ahead_python_and_R_7_0.png)
    



```python
regr.plot("realgdp")
```


    
![png](thierrymoudiki_20240224_ahead_python_and_R_files/thierrymoudiki_20240224_ahead_python_and_R_8_0.png)
    



```python
regr2.plot("realgovt")
```


    
![png](thierrymoudiki_20240224_ahead_python_and_R_files/thierrymoudiki_20240224_ahead_python_and_R_9_0.png)
    



```python
regr2.plot("realgdp")
```


    
![png](thierrymoudiki_20240224_ahead_python_and_R_files/thierrymoudiki_20240224_ahead_python_and_R_10_0.png)
    


# 2 - R version examples


```python
%load_ext rpy2.ipython
```

    The rpy2.ipython extension is already loaded. To reload it, use:
      %reload_ext rpy2.ipython



```r
%%R

options(repos = c(
    techtonique = 'https://techtonique.r-universe.dev',
    CRAN = 'https://cloud.r-project.org'))

install.packages("ahead")
install.packages("ggplot2")
install.packages("forecast")
install.packages("dfoptim")
```


```r
%%R

library(ahead)
library(dfoptim)
```


```python
# Convert to R DataFrame
%R -i df
```

    /usr/local/lib/python3.10/dist-packages/rpy2/robjects/pandas2ri.py:55: FutureWarning: iteritems is deprecated and will be removed in a future version. Use .items instead.
      for name, values in obj.iteritems():



```python
%R df_ts <- as.ts(df)
```


```r
%%R

objective_function <- function(xx)
{
    ahead::loocvridge2f(df_ts,
                        h = 20,
                        type_pi="blockbootstrap",
                        lambda_1=10^xx[1],
                        lambda_2=10^xx[2],
                        show_progress = FALSE,
                        )$loocv
}
start <- proc.time()[3]
(opt <- dfoptim::nmkb(fn=objective_function, lower=c(-10,-10), upper=c(10,10), par=c(0.1, 0.1)))
print(proc.time()[3]-start)
```

    elapsed 
      6.657 



```r
%%R

start <- proc.time()[3]
res <- ahead::ridge2f(df_ts, h = 20,
                      type_pi="blockbootstrap",
                      lambda_1=10^opt$par[1], # 'optimal' parameters
                      lambda_2=10^opt$par[2]) # 'optimal' parameters
print(proc.time()[3]-start)
```

      |======================================================================| 100%
    elapsed 
      0.805 



```r
%%R

par(mfrow=c(2, 2))
plot(res, "realgovt", type = "sims", main="50 predictive simulations \n of realgovt")
plot(res, "realgovt", type = "dist", main="predictive distribution \n of realgovt")
plot(res, "realgdp", type = "sims", main="50 predictive simulations \n of realgdp")
plot(res, "realgdp", type = "dist", main="predictive distribution \n of realgovt")
```


    
![png](thierrymoudiki_20240224_ahead_python_and_R_files/thierrymoudiki_20240224_ahead_python_and_R_19_0.png)
    



```r
%%R

library(ggplot2)
library(forecast)

x <- fdeaths
xreg <- ahead::createtrendseason(x)
(z <- ahead::ridge2f(x, xreg = xreg, h=20))
autoplot(z)
```


    
![png](thierrymoudiki_20240224_ahead_python_and_R_files/thierrymoudiki_20240224_ahead_python_and_R_20_0.png)
    

