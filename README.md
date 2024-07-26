# Portfolio Optimization System

### Project Scope
__Objective:__ Minimize the risk (variance) of a portfolio while maintaining a certain level of expected return.

__Constraints:__ Budget constraints, risk tolerance, investment horizon, etc.



predict price for future for all stocks
it will give you future data - collect this data and save it
then classify each stock based on future data is it long term or short term and then we can apply portfolio optimization tool

### Project Structure
- `backend/`: contains Flask API for portfolio optimization and stock price prediction'
- `data/`: contains data files for stock price prediction
- `models/`: contains trained models for stock price prediction
- `notebooks/`: contains jupyter notebooks for data analysis and model training
- `prototype/`: contains prototype code for portfolio optimization and stock price prediction
- `services/`: contains services for portfolio optimization and stock price prediction
- `utils/`: contains utility functions for data processing and model training
- `main.py`: main file to run the project

## Installation
- Clone the repository
- Install required packages using command: "pip install -r requirements.txt"

## Setup Database
The project uses SQLite database to store stock price data. As GitHub don't allow to store large files, you can download the data from [here](https://drive.google.com/file/d/1bOrcvBZL80MOglPDY8TWdnY8-Dyujwmi/view?usp=sharing).

After downloading the file, Make db folder and place database file in `data/db/` directory, __Very Important__ to place the database file in `data/db/` directory.


## Run Project
- run command: "python3 main.py" from root directory of project.
- Terminal will ask your input for which app to run
  - Option:
    - `blpo`: portfolio optimization,
    - `stock_trend_pred`: stock price prediction
    - `backend`: Flask API
