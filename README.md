![FART](./references/fart.webp)

# FART

FART stands for Financial Analysis & Real-time Trading. It is a project to build a trading agent that can trade cryptocurrency in real-time. The agent will be trained on historical data and will be able to make predictions in terms of buy/sell signals. The agent will then use these predictions to make trades in real-time.

## Motivation

The motivation for this project is to learn more about machine learning and to build a trading agent that can possibly generate revenue.

## Name

The name FART is a abbreviation play on the title "Financial Analysis and Real-time Trading". It is also the name of a gaseous being who appeared in the episode [Mortynight Run
](https://www.imdb.com/title/tt4832254/) of the Rick and Morty series. There is a bit of musical part in the episode, which I really enjoyed. 

## Project Structure

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience

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

This will start the initialization of the trading agent by optionally training the model and then running the agent.

![FART](./references/dashboard.png)

Other commands are:

```bash
make data # to make the dataset
make train # to train the model
```

## Flowchart

The flowchart of the trading agent is as follows:

```mermaid
stateDiagram
      [*] --> Initialize
      Initialize --> Extend_Flowchart
      Extend_Flowchart --> [*]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Rick and Morty](https://www.imdb.com/title/tt2861424/)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Mermaid](https://mermaid-js.github.io/mermaid/#/)
