from pydantic import BaseModel


class BBandsConfig(BaseModel):
    """
    Bollinger Bands configuration

    Attributes
    ----------
    - period (int): Time period for the moving average
    - standard_deviation (int): Standard deviation for the upper and lower
    bands

    """

    period: int = 20
    standard_deviation: int = 2


class EMAConfig(BaseModel):
    """
    Exponential Moving Averages configuration

    Attributes
    ----------
    - fast_period (int): Time period for the fast moving average
    - slow_period (int): Time period for the slow moving average

    """

    fast_period: int = 9
    slow_period: int = 21


class MACDConfig(BaseModel):
    """
    Moving Average Convergence Divergence configuration

    Attributes
    ----------
    - fast_period (int): Time period for the fast moving average
    - slow_period (int): Time period for the slow moving average
    - signal_period (int): Time period for the signal line

    """

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9


class RSIConfig(BaseModel):
    """
    Relative Strength Index configuration

    Attributes
    ----------
    - period (int): Time period for the RSI
    - overbought (int): Overbought level for the RSI
    - oversold (int): Oversold level for the RSI
    - margin (int): Margin between overbought and oversold levels

    """

    period: int = 14
    overbought: int = 70
    oversold: int = 30
    margin: int = 10


class TechnicalIndicatorsConfig(BaseModel):
    """
    Configuration for technical indicators

    Attributes
    ----------
    - bbands (BBandsConfig): Bollinger Bands configuration
    - ema (EMAConfig): Exponential Moving Averages configuration
    - macd (MACDConfig): Moving Average Convergence Divergence configuration
    - rsi (RSIConfig): Relative Strength Index configuration

    """

    bbands: BBandsConfig = BBandsConfig()
    ema: EMAConfig = EMAConfig()
    macd: MACDConfig = MACDConfig()
    rsi: RSIConfig = RSIConfig()
