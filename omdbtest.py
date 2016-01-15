import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import tmdbsimple as tmdb

tmdb.API_KEY = '815963adb42371693b2dc25fd8cfd1c8'


# columns: year, length, budget, rating, action, animation, comedy, drama, documentary, romance, short
data = np.genfromtxt('movies.tab', dtype = "S36,i4,i4,i4,i4,b,b,b,b,b,b,b", comments = '\\', delimiter = '\t', skip_header = 1, usecols = (0,1,2,3,4,17,18,19,20,21,22,23), max_rows = 10)

for film in data:
    title = film[0].decode("utf-8")
    search = tmdb.Search()
    response = search.movie(query = title)
    print(title, ": ")
    print(size(search.results))
    for s in search.results:
        print(s['title'], s['id'], s['release_date'], s['popularity'])
    