# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""DePlot Prompts."""

import enum


class TemplateKey(enum.Enum):
  QA = 'qa'
  POT = 'pot'
  STATISTA_SUMMARY = 'statista_summary'
  PEW_SUMMARY = 'pew_summary'


_INSTRUCTION = 'Read the table below to answer the following questions.'
_POT_INSTRUCTION = ('Read the table below to write code to answer the following'
                    ' questions using the variable ans.')
_TABLE = """Year | Democrats | Republicans | Independents
2004 | 68.1% | 45.0% | 53.0%
2006 | 58.0% | 42.0% | 53.0%
2007 | 59.0% | 38.0% | 45.0%
2009 | 72.0% | 49.0% | 60.0%
2011 | 71.0% | 51.2% | 58.0%
2012 | 70.0% | 48.0% | 53.0%
2013 | 72.0% | 41.0% | 60.0%"""


_SUMMARY_INSTRUCTION = ('Read the statistic below and summarize it in a couple'
                        ' of sentences.')
_PEW_TEMPLATE = f"""{_SUMMARY_INSTRUCTION}

Title: 42% of Facebook users have taken a break from the site in the past year
Activity | Percentage
Any of activities | 74
Adjusted their privacy settings | 54
Taken a break from checking for several weeks or more | 42
Deleted the app from their phone | 26

Summary: Significant shares of Facebook users have taken steps in the past year to reframe their relationship with the social media platform. Just over half of Facebook users ages 18 and older (54%) say they have adjusted their privacy settings in the past 12 months, according to a new Pew Research Center survey. Around four-in-ten (42%) say they have taken a break from checking the platform for a period of several weeks or more, while around a quarter (26%) say they have deleted the Facebook app from their cellphone.

------

{_SUMMARY_INSTRUCTION}"""

_STATISTA_TEMPLATE = f"""{_SUMMARY_INSTRUCTION}

Title: Unemployment rate of those whose main language was not English in England and Wales in 2011 , by gender and proficiency
Characteristic | Male | Female
Could speak English "very well" or "well" | 8.9% | 10.3%
Could speak English "not well" or "not at all" | 10.2% | 15.1%

Summary: This statistic shows the unemployment rate of non-native speakers in England and Wales in 2011 , by gender and English language proficiency . The unemployment rate was greater for those who were not proficient in English than for those who were , and in both categories the unemployment rate was higher for women .

------

{_SUMMARY_INSTRUCTION}

Title: Average number of daily brand posts on Twitter in 2019 , by vertical
Characteristic | Number of brand tweets per day
Media | 9.54
Sports teams | 7.24
Nonprofits | 1.92
Higher ed | 1.44
Tech & software | 1.19
Financial services | 0.88
Influencers | 0.84
Total | 0.77
Home decor | 0.48
Hotels & resorts | 0.44
Retail | 0.41
Alcohol | 0.38
Health & beauty | 0.35
Food & beverage | 0.35
Fashion | 0.34

Summary: This statistic presents the average number of daily brand posts on Twitter in 2019 , broken down by vertical . According to industry data , leading media brands generated an average of 9.54 tweets every day . In contrast , retail brands only tweeted 0.41 times per day on average .

------

{_SUMMARY_INSTRUCTION}

Title: Leading players of soccer team Club AtlÃ©tico River Plate in Argentina as of December 2019 , by market value ( in million U.S. dollars )
Characteristic | Market value in million U.S. dollars
Exequiel Palacios (CM) | 28.5
Nicolás De La Cruz (AM) | 17.1
Santos Borré (CF) | 14.25
Lucas Martínez Quarta (CB) | 14.25
Juan Fernando Quintero (AM) | 11.4
Gonzalo Montiel (RB) | 11.4
Cristian Ferreira (CM) | 11.4
Julián Álvarez (CF) | 11.4

Summary: As of December 2019 , River Plate 's center midfield Exequiel Palacios was the most valuable player in this Argentinian soccer team , with a market value of approximately 28.5 million U.S. dollars , followed by the attacking midfield Nicolás De La Cruz with a market value of around 17.1 million U.S. dollars .

------

{_SUMMARY_INSTRUCTION}

Title: Turnover of the furniture manufacturing industry in Italy from 2008 to 2017 ( in million euros )
Characteristic | Annual turnover in million euros
2017 | 22300.6
2016 | 21647.3
2015 | 20759.3
2014 | 19861.9
2013 | 19536.6
2012 | 19494.8
2011 | 20255.6
2010 | 21566.9
2009 | 21471.5
2008* | 25811.2

Summary: This statistic shows the annual turnover from the manufacture of furniture in Italy from 2008 to 2017 . In 2017 , the manufacturing of furniture produced a turnover of approximately 22.3 billion euros .

------

{_SUMMARY_INSTRUCTION}

Title: Callaway Golf 's sales share worldwide in 2019 , by product category
Characteristic | Share of sales Golf
clubs | 45.2%
Apparel | 24.1%
Gear, accessories & other | 18.3%
Golf balls | 12.4%

Summary: In 2019 , 45.2 percent of Callaway Golf 's sales came from golf clubs . The company had golf club sales of 768.3 million U.S. dollars that year .

------

{_SUMMARY_INSTRUCTION}"""


def _add_markup(table):
  parts = [p.strip() for p in table.splitlines(keepends=False)]
  if parts[0].startswith('TITLE'):
    result = f"Title: {parts[0].split(' | ')[1].strip()}\n"
    rows = parts[1:]
  else:
    result = ''
    rows = parts
  prefixes = ['Header: '] + [f'Row {i+1}: ' for i in range(len(rows) - 1)]
  return result + '\n'.join(prefix + row for prefix, row in zip(prefixes, rows))


def _skip_title(table):
  return '\n'.join(part for part in table.splitlines(keepends=False)
                   if not part.startswith('TITLE'))


_TEMPLATE = f"""{_INSTRUCTION}

{_add_markup(_TABLE)}

Q: In which year republicans have the lowest favor rate?
A: Let's find the column of republicans. Then let's extract the favor rates, they [45.0, 42.0, 38.0, 49.0, 51.2, 48.0, 41.0]. The smallest number is 38.0, that's Row 3.  Row 3 is year 2007. The answer is 2007.

Q: What is the sum of Democrats' favor rates of 2004, 2012, and 2013?
A: Let's find the rows of years 2004, 2012, and 2013. We find Row 1, 6, 7. The favor dates of Demoncrats on that 3 rows are 68.1, 70.0, and 72.0. 68.1+70.0+72=210.1. The answer is 210.1.

Q: By how many points do Independents surpass Republicans in the year of 2011?
A: Let's find the row with year = 2011. We find Row 5. We extract Independents and Republicans' numbers. They are 58.0 and 51.2. 58.0-51.2=6.8. The answer is 6.8.

Q: Which group has the overall worst performance?
A: Let's sample a couple of years. In Row 1, year 2004, we find Republicans having the lowest favor rate 45.0 (since 45.0<68.1, 45.0<53.0). In year 2006, Row 2, we find Republicans having the lowest favor rate 42.0 (42.0<58.0, 42.0<53.0). The trend continues to other years. The answer is Republicans.

Q: Which party has the second highest favor rates in 2007?
A: Let's find the row of year 2007, that's Row 3. Let's extract the numbers on Row 3: [59.0, 38.0, 45.0]. 45.0 is the second highest. 45.0 is the number of Independents. The answer is Independents.


{_INSTRUCTION}"""


_POT_TEMPLATE = f"""{_POT_INSTRUCTION}

{_add_markup(_TABLE)}

Q: What was the average difference in approval rates between democrats and republicans in 2006 and 2007?
#Python
democrats_2006 = 58.0
republicans_2006 = 42.0
# The difference between A and B is A - B which may be negative
difference_2006 = democrats_2006 - republicans_2006
democrats_2007 = 59.0
republicans_2007 = 38.0
difference_2007 = democrats_2007 - republicans_2007
ans = (difference_2006 + difference_2007) / 2

Q: What is the average of Democrats' favor rates of 2004, 2012, and 2013?
#Python
# Years 2004, 2012, and 2013  correspond to rows 1, 6 and 7.
democrats_2004 = 68.1
democrats_2012 = 70.0
democrats_2013 = 72.0
ans = (democrats_2004 + democrats_2012 + democrats_2013) / 3

Q: Which party had less than 50% approval rate in 2013?
#Python
# year 2013 corresponds to row 7. Numbers on row 7 are [72.0, 41.0, 60.0]
# Republicans are the only with less than 50.
ans = "Republicans"

Q: What is percentage of relative increase in approval rate for democrats from 2012 to 2013?
#Python
# year 2012 and 2013 correspond to rows 6 and 7.
# Numbers of democrats on row 6 are 70.0
democrats_2012 = 70.0
# Numbers of democrats on row 7 are 72.0
democrats_2013 = 72.0
ans = 100 * (democrats_2013 - democrats_2012) / democrats_2012

Q: What is the difference between republicans in 2011 and democrats in 2006?
#Python
# year = 2011 corresponds to row 5 and the republicans had a 51.2 rate
republicans_2011 = 51.2
# year = 2006 corresponds to row 2 and the democrats had a 58.0 rate
democrats_2006 = 58.0
# The difference between A and B is A - B which may be negative
ans = republicans_2011 - democrats_2006


{_POT_INSTRUCTION}"""


def get_template(template_key):
  """Returns a template given the key which identifies it."""
  if template_key == TemplateKey.QA:
    return _TEMPLATE
  if template_key == TemplateKey.POT:
    return _POT_TEMPLATE
  elif template_key == TemplateKey.STATISTA_SUMMARY:
    return _STATISTA_TEMPLATE
  elif template_key == TemplateKey.PEW_SUMMARY:
    return _PEW_TEMPLATE
  else:
    raise ValueError(f'Invalid template key {template_key}')


def build_prompt(template_key, table, question):
  """Builds a prompt given a table, question and a template identifier."""
  template = get_template(template_key)
  if template_key == TemplateKey.QA:
    return f"""{template}

{_add_markup(table)}

Q: {question}
A: """
  elif template_key == TemplateKey.POT:
    return f"""{template}

{_add_markup(table)}

Q: {question}
"""

  return f"""{template}

Title: {question}
{_skip_title(table)}

Summary: """