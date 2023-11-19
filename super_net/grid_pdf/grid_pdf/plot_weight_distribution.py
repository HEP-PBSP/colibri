import json
import matplotlib.pyplot as plt
from ultranest.plot import cornerplot

import corner
import numpy as np


ns_file = "results.json"


with open(ns_file, 'r') as file:
    ns_data = json.loads(json.load(file))




# Create a corner plot
figure = cornerplot(ns_data) 

# Show the plot
plt.savefig("ns_weights_distribution.pdf")
