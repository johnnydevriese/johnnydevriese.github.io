---
categories: kaggle machine-learning
date: '2025-03-04'
layout: post
slug: virigina-tech-bicycle-helmets
title: Virigina Tech Bicycle Helmets
---

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
df = pd.read_csv('/kaggle/input/virginia-tech-bicycle-helmet-ratings/full_bicycle_helmet_ratings.csv')
```


```python
df
```


```python
pip install plotly
```


```python
import pandas as pd
import plotly.express as px

# Load the data
# df = pd.read_csv('helmet_data.csv')

# Create a more visually appealing scatter plot with hover-only labels
fig = px.scatter(
    df, 
    x='Cost ($)', 
    y='Score',
    color='Score',  # Color points by score for visual appeal
    size='Score',   # Vary point size by score
    hover_name='Helmet',  # Display helmet name on hover
    hover_data={
        'Helmet': False,  # Hide redundant label
        'Cost ($)': True,
        'Score': True
    },
    title='Bicycle Helmet Ratings: Price vs. Safety Score',
    labels={
        'Cost ($)': 'Price (USD)', 
        'Score': 'Safety Score'
    },
    color_continuous_scale='viridis',  # Add a nice color gradient
    opacity=0.8
)

# Improve the layout and styling
fig.update_traces(
    marker=dict(
        line=dict(width=1, color='DarkSlateGrey')  # Add a subtle border
    )
)

# Enhance the overall appearance
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font_family='Arial',
    font_size=12,
    title_font_size=18,
    title_x=0.5,  # Center the title
    xaxis=dict(
        showgrid=True, 
        gridcolor='lightgrey',
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True, 
        gridcolor='lightgrey',
        zeroline=False
    ),
    coloraxis_colorbar=dict(
        title='Safety<br>Score'
    )
)

# Add a trendline (optional)
fig.update_layout(
    shapes=[
        dict(
            type='line',
            xref='x', yref='y',
            x0=df['Cost ($)'].min(),
            y0=df['Score'].min(),
            x1=df['Cost ($)'].max(),
            y1=df['Score'].max(),
            opacity=0.3,
            line=dict(
                color='red',
                width=2,
                dash='dash',
            )
        )
    ]
)

fig.show(renderer='notebook')

# Show the plot
fig.show()
```
