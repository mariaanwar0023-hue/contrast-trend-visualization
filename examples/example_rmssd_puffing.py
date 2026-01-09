import numpy as np
import matplotlib.pyplot as plt
from src.trend_plot import plot_contrast_trend

# ---------------------------------------------------------
# Synthetic example values (illustrative only)
# These do NOT correspond to any real participant or study
# ---------------------------------------------------------

# Example estimated marginal means
means = np.array([0.45, -0.10, -0.85])

# Corresponding 95% confidence intervals
ci_lower = np.array([-0.30, -0.85, -1.60])
ci_upper = np.array([1.20, 0.65, -0.10])

# Example linear contrast statistics (synthetic)
lin_stats = {
    "estimate": -0.62,
    "se": 0.41,
    "t": -1.52,
    "p": 0.14,
    "df": 30,
    "ci_lower": -1.48,
    "ci_upper": 0.24,
}

plot_contrast_trend(
    means=means,
    ci_lower=ci_lower,
    ci_upper=ci_upper,
    doses_original=[0.5, 3.0, 12.0],     # altered doses
    dose_labels=["Low", "Medium", "High"],
    variable_name="Example Outcome",
    lin_coeffs=[-1, 0, 1],               # simple illustrative contrast
    lin_stats=lin_stats,
)

plt.show()
