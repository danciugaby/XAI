import numpy as np
from cfel import CFEL

# Generate counterfactual explanations for the selected instance
cfel = CFEL(model, x, target_class=0)
cf_explanation = cfel.generate_counterfactual()

