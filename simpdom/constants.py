# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All the constant varaibales that can be reused in many files.
"""


VERTICAL_WEBSITES = {
    "auto": [
        "msn", "aol", "kbb", "cars", "yahoo", "autoweb", "autobytel",
        "automotive", "carquotes", "motortrend"
    ],
    "book": [
        "abebooks", "amazon", "barnesandnoble", "bookdepository",
        "booksamillion", "borders", "buy", "christianbook", "deepdiscount",
        "waterstones"
    ],
    "camera": [
        "amazon", "beachaudio", "buy", "compsource", "ecost", "jr", "newegg",
        "onsale", "pcnation", "thenerds"
    ],
    "job": [
        "careerbuilder", "dice", "hotjobs", "job", "jobcircle", "jobtarget",
        "monster", "nettemps", "rightitjobs", "techcentric"
    ],
    "movie": [
        "allmovie", "amctv", "boxofficemojo", "hollywood", "iheartmovies",
        "imdb", "metacritic", "msn", "rottentomatoes", "yahoo"
    ],
    "nbaplayer": [
        "espn", "fanhouse", "foxsports", "msnca", "nba", "si", "slam",
        "usatoday", "wiki", "yahoo"
    ],
    "restaurant": [
        "fodors", "frommers", "gayot", "opentable", "pickarestaurant",
        "restaurantica", "tripadvisor", "urbanspoon", "usdiners", "zagat"
    ],
    "university": [
        "collegeboard", "collegenavigator", "collegeprowler", "collegetoolkit",
        "ecampustours", "embark", "matchcollege", "princetonreview",
        "studentaid", "usnews"
    ]
}

ATTRIBUTES = {
    "auto": ["model", "price", "engine", "fuel_economy"],
    "book": ["title", "author", "isbn_13", "publisher", "publication_date"],
    "camera": ["model", "price", "manufacturer"],
    "job": ["title", "company", "location", "date_posted"],
    "movie": ["title", "director", "genre", "mpaa_rating"],
    "nbaplayer": ["name", "team", "height", "weight"],
    "restaurant": ["name", "address", "phone", "cuisine"],
    "university": ["name", "phone", "website", "type"]
}

ATTRIBUTES_PAD = {
    "auto": ["model", "price", "engine", "fuel_economy", "<PAD>"],
    "book": ["title", "author", "isbn_13", "publisher", "publication_date"],
    "camera": ["model", "price", "manufacturer", "<PAD>", "<PAD>"],
    "job": ["title", "company", "location", "date_posted", "<PAD>"],
    "movie": ["title", "director", "genre", "mpaa_rating", "<PAD>"],
    "nbaplayer": ["name", "team", "height", "weight", "<PAD>"],
    "restaurant": ["name", "address", "phone", "cuisine", "<PAD>"],
    "university": ["name", "phone", "website", "type", "<PAD>"]
}

ATTRIBUTES_PLUS_NONE = {
    "auto": ["engine", "fuel_economy", "model", "none", "price"],
    "book": [
        "author", "isbn_13", "none", "publication_date", "publisher", "title"
    ],
    "camera": ["manufacturer", "model", "none", "price"],
    "job": ["company", "date_posted", "location", "none", "title"],
    "movie": ["director", "genre", "mpaa_rating", "none", "title"],
    "nbaplayer": ["height", "name", "none", "team", "weight"],
    "restaurant": ["address", "cuisine", "name", "none", "phone"],
    "university": ["name", "none", "phone", "type", "website"]
}
