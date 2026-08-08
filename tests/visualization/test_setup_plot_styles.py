import matplotlib.pyplot as plt

from fart.visualization.plot_styles import apply_plot_styles


def test_apply_plot_styles_applies_tradingview_rcparams() -> None:
    plt.rcdefaults()

    apply_plot_styles()

    assert plt.rcParams["axes.titlecolor"] == "red"
    assert plt.rcParams["axes.edgecolor"] == "grey"
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["grid.linestyle"] == "--"
    assert plt.rcParams["figure.facecolor"] == "white"

    plt.rcdefaults()
