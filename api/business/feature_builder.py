import numpy as np
from api.schemas import BusinessInputRequest


class BusinessFeatureBuilder:
    """
    Builds year-independent, scenario-based features
    from available engineered climate, soil, and vegetation data.
    """

    def build(self, req: BusinessInputRequest) -> np.ndarray:
        features = [
            # Climate
            req.climate.avg_temperature,      # T2M
            req.climate.max_temperature,      # T2M_MAX
            req.climate.min_temperature,      # T2M_MIN
            req.climate.total_rainfall,       # PRECTOTCORR
            req.climate.solar_radiation,      # ALLSKY_SFC_SW_DWN
            req.climate.relative_humidity,    # RH2M
            req.climate.wind_speed,            # WS2M

            # Soil
            req.soil.soil_organic_carbon,     # Mean_SOC

            # Vegetation
            req.vegetation.ndvi_early,        # NDVI_SeasonalMean
        ]

        return np.array(features).reshape(1, -1)
