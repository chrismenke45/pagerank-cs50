from pagerank import transition_model
import random

corpus = {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}
page = "1.html"
damping_factor = 0.85

print(transition_model(corpus, page, damping_factor))