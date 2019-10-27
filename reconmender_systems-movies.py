import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating = 4.0)
#min_ratings sets a threshold for collecting the data
#means we only collect data above 4.0 ratings

print(repr(data['train']))
print(repr(data['test']))
