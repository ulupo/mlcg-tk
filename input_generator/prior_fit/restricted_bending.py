import torch
import numpy as np
from typing import Dict
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid


def restricted_quartic_angle(x, a, b, c, d, k, V0):
    """Quartic angle potential with repulsive 1/sin^2 term to avoid
    singularities at 0 and pi angles.
    
    .. math:
        V(\theta) = a cos(\theta)^4 + b cos(\theta)^3 + c cos(\theta)^2 + d cos(\theta) 
        + \frac{k}{\sin^2(\theta)} + V0

    """

    quart = a * torch.pow(x, 4) + b * torch.pow(x,3) + c * torch.pow(x,2) + d * (x)
    raw_ang = torch.acos(x)
    rep = k / torch.pow(torch.sin(raw_ang),2) 
    V = quart + rep + V0

    return V


def fit_rb_from_potential_estimates(
    bin_centers_nz: torch.Tensor, dG_nz: torch.Tensor
) -> Dict:
    r"""if bounds is provided a constrain on the derivative of the
    potential is considered so that the derivative is lower than bound[0]
    and greather than bound[1]"""

    # remove noise by discarding signals
    integral = torch.tensor(
        float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
    )

    mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)
    try:
        popt, _ = curve_fit(
            restricted_quartic_angle,
            bin_centers_nz[mask],
            dG_nz[mask],
            # Here initial guess for k must be small but not matching the lower bound
            p0=[1, 0, 0, 0, 1e-4,  torch.argmin(dG_nz[mask])],
            # Here we set a to be positive to ensure postive curvature 
            # and k positive to avoid -inf at 0 and pi values
            bounds=(
                (0, -np.inf, -np.inf, -np.inf, 1e-5, -np.inf),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
            ),
            maxfev=5000,
        )
        stat = {
            "a": popt[0],
            "b": popt[1],
            "c": popt[2],
            "d": popt[3],
            "k": popt[4],
            "V0": popt[5],
        }
    except:
        print(f"failed to fit potential estimate for DoubleAngle")
        stat = {
            "a": torch.tensor(float("nan")),
            "b": torch.tensor(float("nan")),
            "c": torch.tensor(float("nan")),
            "d": torch.tensor(float("nan")),
            "k": torch.tensor(float("nan")),
            "V0": torch.tensor(float("nan")),
        }
    return stat