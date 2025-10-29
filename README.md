# Energy Forecast Dashboard

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

The Energy Forecast Dashboard integrates real-time weather observations with energy market data to provide short-term energy demand forecasts for 30 UK cities.  
It leverages a lightweight **stacking regression ensemble** combining weather-driven and simulated demand patterns.

## Features

- Real-time weather data from **UK Met Office API** with **Open-Meteo Archive** as fallback.
- Energy market insights including **spot prices** and **wind generation**.
- Interactive plots powered by **Plotly**.
- Hourly detailed forecast table with temperature, wind speed, demand, and market data.
- Lightweight, modular, and easily extensible stacking regression model using **XGBoost, LightGBM, and Random Forest**.

## Installation

```bash
pip install -r requirements.txt
