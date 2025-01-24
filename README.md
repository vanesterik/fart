![FART](./references/fart.webp)

# FART

**FART** stands for **F**inancial **A**nalysis & **R**eal-time **T**rading. It is a project to build a trading agent that can trade cryptocurrency in real-time. The agent will be trained on historical data and will be able to make predictions in terms of buy/sell signals. The agent will then use these predictions to make trades in real-time.

## Motivation

The motivation for this project is to learn more about machine learning and to build a trading agent that can possibly generate revenue.

## Name

The name **FART** is a abbreviation play on the title "Financial Analysis and Real-time Trading". It is also the name of a gaseous being who appeared in the episode [Mortynight Run
](https://www.imdb.com/title/tt4832254/) of the Rick and Morty series. There is a bit of musical part in the episode, which I really enjoyed. The name **FART** is a homage to that episode.

## Installation

To install the project, you can use the following commands:

```bash
pdm install
```

## Usage

To use the project, you can use the following commands:

```bash
make run # to run the trading agent
```

The `run` command will start the initialization of the trading agent. The trading agent is based upon a loop specified by the interval passed as an argument to the `run` command (default is 30m). The agent will trigger a function each loop iteration to check the latest candle data.  
With the candle data the agent will make a prediction and place an order based upon the prediction. The agent will then wait for the next interval to repeat the process.

The agent can be terminated by pressing the <kbd>Escape</kbd> key. The agent can also be paused and resumed by toggling the <kbd>space</kbd> key.

By pressing the <kbd>Tab</kbd> or <kbd>Enter</kbd> key, the agent will forcibly place an order to switch position regardless of an potential prediction.

![FART](./references/dashboard.png)

Additional commands are:

```bash
make data # to make the dataset
make train # to train the model
```

These commands are used internally by the `run` command and are not necessary to run executed separately. But are listed in order to show the full extent of the project.


## State Machine Diagram

In order to get a better understanding of the trading agent, a state machine diagram is provided. The state machine diagram shows the different states the trading agent can be in and the transitions between these states:

```mermaid
---
config:
  layout: elk
---
stateDiagram
   state initialization_condition <<choice>>
   state trade_signal_condition <<choice>>
   [*] --> initialization_condition: EVALUATE_INITIALIZATION
   initialization_condition --> collecting_data: INITIALIZE
   initialization_condition --> idle: SKIP_INITIALIZATION
   collecting_data --> preprocessing_data
   preprocessing_data --> training_model
   training_model --> idle
   idle --> predicting_trade_signal: RECEIVE_CANDLE_DATA
   predicting_trade_signal --> trade_signal_condition: EVALUATE_PREDICTION
   trade_signal_condition --> placing_order: SWITCH_POSITION
   trade_signal_condition --> idle: MAINTAIN_POSITION
   idle --> placing_order: FORCE_POSITION_SWITCH
   idle --> pausing: PAUSE_PROGRAM
   pausing --> idle: RESUME_PROGRAM
   placing_order --> idle
   idle --> terminating: TERMINATE_PROGRAM
   terminating --> [*]
```

## Project Structure

The project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience

```
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`.
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data               <- Data from third party sources.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering), the creator's initials, and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting.
    │
    ├── tests              <- Unit tests of various modules and functions.
    │
    ├── pyproject.toml     <- The project file for reproducing the analysis environment, e.g. generated with `pdm init`.
    │                         
    └── src/fart           <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module.
        │
        ├── data           <- Scripts to download or generate data.
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make predictions.
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations.
            └── visualize.py
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Rick and Morty](https://www.imdb.com/title/tt2861424/)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Mermaid](https://mermaid-js.github.io/mermaid/#/)
