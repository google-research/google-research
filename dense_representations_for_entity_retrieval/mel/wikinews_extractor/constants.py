# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# -*- coding: utf-8 -*-
"""Literals used for heuristic processing of wikinews links and articles."""

# Allow-list of Wikipedia Interlanguage prefixes.
# Source: https://meta.wikimedia.org/wiki/Special:Interwiki
# We extract a mention for any link that points to a page in one of these
# Wikipedia editions, which we can map to WikiData QIDs. Note this is different
# from the broader Interwiki prefixes which contain arbitrary things like
# 'wookieepedie' -> 'starwars.wikia.com/wiki'.
# TODO(jabot): Look into the handful of language code # aliases listed among the
WIKI_LANG_CODES = (
    "aa",
    "ab",
    "ace",
    "ady",
    "af",
    "ak",
    "als",
    "am",
    "an",
    "ang",
    "ar",
    "arc",
    "arz",
    "as",
    "ast",
    "atj",
    "av",
    "ay",
    "az",
    "azb",
    "ba",
    "ban",
    "bar",
    "bat-smg",
    "bcl",
    "be",
    "be-tarask",
    "be-x-old",
    "bg",
    "bh",
    "bi",
    "bjn",
    "bm",
    "bn",
    "bo",
    "bpy",
    "br",
    "bs",
    "bug",
    "bxr",
    "ca",
    "cbk-zam",
    "cdo",
    "ce",
    "ceb",
    "ch",
    "cho",
    "chr",
    "chy",
    "ckb",
    "co",
    "cr",
    "crh",
    "cs",
    "csb",
    "cu",
    "cv",
    "cy",
    "da",
    "de",
    "din",
    "diq",
    "dsb",
    "dty",
    "dv",
    "dz",
    "ee",
    "egl",
    "el",
    "eml",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "ext",
    "fa",
    "ff",
    "fi",
    "fiu-vro",
    "fj",
    "fo",
    "fr",
    "frp",
    "frr",
    "fur",
    "fy",
    "ga",
    "gag",
    "gan",
    "gcr",
    "gd",
    "gl",
    "glk",
    "gn",
    "gom",
    "gor",
    "got",
    "gsw",
    "gu",
    "gv",
    "ha",
    "hak",
    "haw",
    "he",
    "hi",
    "hif",
    "ho",
    "hr",
    "hsb",
    "ht",
    "hu",
    "hy",
    "hyw",
    "hz",
    "ia",
    "id",
    "ie",
    "ig",
    "ii",
    "ik",
    "ilo",
    "inh",
    "io",
    "is",
    "it",
    "iu",
    "ja",
    "jam",
    "jbo",
    "jv",
    "ka",
    "kaa",
    "kab",
    "kbd",
    "kbp",
    "kg",
    "ki",
    "kj",
    "kk",
    "kl",
    "km",
    "kn",
    "ko",
    "koi",
    "kr",
    "krc",
    "ks",
    "ksh",
    "ku",
    "kv",
    "kw",
    "ky",
    "la",
    "lad",
    "lb",
    "lbe",
    "lez",
    "lfn",
    "lg",
    "li",
    "lij",
    "lmo",
    "ln",
    "lo",
    "lrc",
    "lt",
    "ltg",
    "lv",
    "lzh",
    "mai",
    "map-bms",
    "mdf",
    "mg",
    "mh",
    "mhr",
    "mi",
    "min",
    "mk",
    "ml",
    "mn",
    "mnw",
    "mo",
    "mr",
    "mrj",
    "ms",
    "mt",
    "mus",
    "mwl",
    "my",
    "myv",
    "mzn",
    "na",
    "nah",
    "nan",
    "nap",
    "nb",
    "nds",
    "nds-nl",
    "ne",
    "new",
    "ng",
    "nl",
    "nn",
    "no",
    "nov",
    "nqo",
    "nrm",
    "nso",
    "nv",
    "ny",
    "oc",
    "olo",
    "om",
    "or",
    "os",
    "pa",
    "pag",
    "pam",
    "pap",
    "pcd",
    "pdc",
    "pfl",
    "pi",
    "pih",
    "pl",
    "pms",
    "pnb",
    "pnt",
    "ps",
    "pt",
    "qu",
    "rm",
    "rmy",
    "rn",
    "ro",
    "roa-rup",
    "roa-tara",
    "ru",
    "rue",
    "rup",
    "rw",
    "sa",
    "sah",
    "sat",
    "sc",
    "scn",
    "sco",
    "sd",
    "se",
    "sg",
    "sgs",
    "sh",
    "shn",
    "si",
    "simple",
    "sk",
    "sl",
    "sm",
    "sn",
    "so",
    "sq",
    "sr",
    "srn",
    "ss",
    "st",
    "stq",
    "su",
    "sv",
    "sw",
    "szl",
    "szy",
    "ta",
    "tcy",
    "te",
    "tet",
    "tg",
    "th",
    "ti",
    "tk",
    "tl",
    "tn",
    "to",
    "tpi",
    "tr",
    "ts",
    "tt",
    "tum",
    "tw",
    "ty",
    "tyv",
    "udm",
    "ug",
    "uk",
    "ur",
    "uz",
    "ve",
    "vec",
    "vep",
    "vi",
    "vls",
    "vo",
    "vro",
    "wa",
    "war",
    "wo",
    "wuu",
    "xal",
    "xh",
    "xmf",
    "yi",
    "yo",
    "yue",
    "za",
    "zea",
    "zh",
    "zh-classical",
    "zh-cn",
    "zh-min-nan",
    "zh-tw",
    "zh-yue",
    "zu",
)

# Manually curated list of headings that signal the end of article content.
END_OF_CONTENT_HEADINGS = (
    ## German (de) ############################################################
    "externe links",
    "referenzen",  # references
    "themenverwandte artikel",  # related articles
    "quellen",  # sources
    "quelle",  # source
    "dokument",  # document
    # 2010-10-17_00_Bedeutender:
    "hintergrundinformationen bzw. ansprechpartner",
    "links",
    "weblink",
    "weblink:",
    "weblinks",
    "video",
    "videos",
    "videolinks",
    ## Turkish (tr) ############################################################
    "kaynakça",  # references
    "kaynaklar",  # resources
    "kaynak",  # source
    "dış bağlantılar",  # external links
    "ilgili haberler",  # related news
    "iç bağlantılar",  # internal links
    ## Arabic (ar) #############################################################
    "المصادر",  # sources (definite)
    "مصادر",  # sources
    "اقرأ أيضا",  # also read
    "نظر أيضاً",  # see also
    ## Serbian (sr) ############################################################
    "извор",  # source
    "izvor",  # source
    "најновије вести",  # latest news
    ## Tamil (ta) ##############################################################
    "மூலம்",  # source
    "தொடர்புள்ள செய்திகள்    ",  # related news
    "தொடர்புள்ள செய்தி  ",  # related news
    "சான்றுகள்",  # references
    ## Japanese (ja) ###########################################################
    "出典",  # source
    "情報源",  # source
    "関連記事",  # related articles
    "外部リンク",  # external links
    ## Spanish (es) ############################################################
    "fuentes",  # sources
    "fuente",  # source
    "noticias relacionadas",  # related news
    "noticia relacionada",  # related news
    "enlaces externos",  # external links
    "enlace externo",  # external link
    "ver también",  # see also
    "véase también",  # see also
    ## Farsi (fa) ##############################################################
    "منبع",  # source
    "ﻢﻧﺎﺒﻋ",  # sources
    "آخرین اخبار",  # latest news
    "پیوند به بیرون",  # references
    "منابع",  # resources/sources(?)
    "جستارهای وابسته",  # related
    ## English (en) ############################################################
    "external link",
    "external links",
    "footer",
    "other news",
    "reactions",
    "reference",
    "references",
    "related news",
    "related stories",
    "related wikinews",
    "see also",
    "sister links",
    "source",
    "sources",
    ## Catalan (ca) ############################################################
    "enllaç extern",  # external link
    "enllaços externs",  # external links
    "font",  # source
    "fontes",  # sources
    "fonts",  # sources
    "notícies relacionades",  # related news
    "referències",  # references
    "vegeu també",  # see also
    ## Czech (cs) ############################################################
    "zdroje",  # resources (references)
    "zdroj",  # resource (reference)
    "externí odkazy",  # external links
    "související zprávy",  # related news
    "odkazy",  # links
    ## Polish (pl) ############################################################
    "linki zewnętrzne",  # external links
    "pokrewne",  #  related
    "przypisy",  # footnotes
    "reakcje",  #  reactions
    "zobacz także",  # see also
    "zobacz też",  #  see also
    "zobacz też:",  # see also
    "źródła",  # references
    "źródła:",  # sources
    "źródło",  # source/reference
    "źródło:",  # source
    ## Romanian (ro) ###########################################################
    "a se vedea și",  #  see also
    "legături externe",  #  external links
    "referințe",  # references
    "sursa",  #  source
    "surse articol",  #  article source
    "surse",  #   sources
    "sursă",  #  source
    "vezi și",  #  see also
    ## Swedish (sv) ###########################################################
    "referenser",  # references
    "läs mer",  # read more
    "publiceringar",  #  publications
    "källa",  #  source
    "källor",  #  sources
    #"senaste nyheterna", #  latest news ??
    "relaterade nyhetsartiklar",  # related news articles
    #"huvudnyheter", #  main news ??
    #"tidigare artiklar", #  previous articles
    "externa länkar",  # external links
    "extern länk",  # external link
    "se även",  # see also
    "fler artiklar",  #  more articles
    "tidigare nyheter",  # previous news
    #"senaste nyheter", # latest news
    "relaterade nyheter",  # related news
    "mer information",  #  more information
    "ämnen",  #  topics
    "vidare läsning",  #  further reading
    "andra nyheter i ämnet",  # other news on subject
    ## Ukrainian (uk) ##########################################################
    "відео",  #  videos
    "джерела",  # sources
    "джерело",  #  source
    "див. також",  #  see also
    "посилання",  #  references
    "примітки",  #  notes
    "цікаві посилання",  #  interesting links
)
# Block-list of term prefixes indicating a wiki-link should not be taken as an
# entity mention. Link targets are matched against these.  Organized by semantic
# type (e.g. "image", "category"), and then by language:
#   [en, de, tr, ar, sr, ta, ja, es, fa, ca, cs, pl, ro, sv, uk],
# some of which may be absent.
LINK_PREFIX_BLOCKLIST = (
    ## IMAGE ###################################################################
    "image",
    "bild",
    "resim",
    # no 'ar' seen yet
    "слика",
    # no 'ta' seen yet
    "画像",
    "imagen",
    "تصویر",
    "grafika",
    "imagine",
    "зображення",
    ## FILE ###################################################################
    "file",
    "datei",
    "dosya",
    "ملف",
    "Датотека",
    "படிமம்",
    "ファイル",
    "archivo",
    "پرونده",
    "fitxer",
    "soubor",
    "plik",
    "fișier",
    "fil",
    "файл",
    ## CATEGORY ################################################################
    "category",
    "kategorie",
    "kategori",
    "تصنيف",
    "категорија",
    # "பகுப்பு ",
    # above value, but raw repr to workaround text editor issues:
    b"\xe0\xae\xaa\xe0\xae\x95\xe0\xaf\x81\xe0\xae\xaa\xe0\xaf\x8d\xe0\xae\xaa\xe0\xaf\x81"
    .decode(),
    # "カテゴリ", # 'ja' just uses "category" in the link prefix
    "categoría",
    "رده",
    "kategoria",
    "categorie",
)

# Block-list of anchor-text patterns for which to skip the text and link
# entirely.
_ANCHOR_EXCLUDE_PATTERNS = (
    # Left-over image markup.
    # TODO(jabot): Check whether this needs to be expanded to more patterns.
    r"\bthumb\|",
    r"\bnáhled\|",

    # Specific issue in Catalan where this template text gets rendered by
    # wikiextractor in the wiki_doc.
    "Per escriure, editar, començar o visualitzar altres articles sobre",
)
