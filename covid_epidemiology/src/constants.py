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

"""Constants used for fit and forecast."""

import os

# GCP Project
PROJECT_ID_MODEL_TRAINING = "covid-forecasting-272503"

# GCP Project in which what-if inference runs are taking place
# Default to the training project but allow to be overridden by env var
PROJECT_ID_WHAT_IF_INFERENCE = os.environ.get("PROJECT_ID_WHAT_IF_INFERENCE",
                                              PROJECT_ID_MODEL_TRAINING)

# Names of environmental variables that will be transferred from the local
# environment into the KFP containers if they are defined.
CONTAINER_ENV_VARIABLES = ["PROJECT_ID_WHAT_IF_INFERENCE"]

# Where possible, dates should be in this format:
DATE_FORMAT = "%Y-%m-%d"

# BigQuery Public Datasets
PUBLIC_PROJECT = "bigquery-public-data"
PUBLIC_GEO_DATASET = "geo_us_boundaries"
PUBLIC_GEO_COUNTY_NEIGHBORS_TABLE_NAME = "adjacent_counties"
PUBLIC_GEO_STATE_NEIGHBORS_TABLE_NAME = "adjacent_states"
GEO_COUNTY_FIPS_CODE_COLUMN = "county_fips_code"
GEO_COUNTY_NAME_COLUMN = "county"
GEO_STATE_FIPS_CODE_COLUMN = "state_fips_code"
GEO_STATE_NAME_COLUMN = "state"
GEO_NEIGHBORS_COLUMN = "neighbors_fips_code"

# Production dataset
PROD_GT_DATASET = "covid_prod_features"
# Comparison Models GT Dataset
COMPARISONS_GT_DATASET = "covid_comparisons_features"
# Simulator output dataset
SIMULATED_GT_DATASET = "covid_simulated_features"

# What-If Override CSV column names
WHAT_IF_CSV_DATE_COLUMN_NAME = "Date"
WHAT_IF_CSV_FORECAST_STARTING_DATE_STRING = "FORECAST STARTING DATE"
WHAT_IF_SCENARIO_COLUMNS = [
    "Date", "State", "Mobility", "NPI - Restaurants",
    "NPI - Non-Essential Businesses", "NPI - Movement", "NPI - Schools",
    "NPI - Emergency", "NPI - Gathering", "NPI - Masks", "Testing Availability",
    "ICU Beds", "Vaccine Ratio", "Vaccine Effectiveness"
]

# Dataset to which to export What-If inferences
WHAT_IF_EXPORT_DATASET = "what_if"
WHAT_IF_METADATA_TABLE = (
    f"{PROJECT_ID_WHAT_IF_INFERENCE}.{WHAT_IF_EXPORT_DATASET}.metadata")

# Table keys
COUNTRY_COLUMN = "country_code"
DATE_COLUMN = "dt"
STATE_CODE_COLUMN = "state_code"
COUNTY_NAME_COLUMN = "county_name"
FEATURE_NAME_COLUMN = "feature_name"
FEATURE_VALUE_COLUMN = "feature_value"
VERSION_COLUMN = "version"
POPULATION_COLUMN = "midyear_population"
GEO_ID_COLUMN = "geo_id"
FEATURE_MODIFIER_COLUMN = "feature_modifier"

# Stable Data Version (Stable = latest data version the model team certifies)
STABLE_VERSION = "2020-08-13 17:39:36 UTC"

# Date from which we start reading features. This is inclusive.
# Note on version "2020-06-19 12:52:04 UTC" this excluded a single confirmed
#   case in the NYT dataset but nothing else.
STARTING_DATE = "2020-01-22"

# Dates for the prospective period for our paper submissiom.
PROSPECTIVE_START_DATE = "2020-11-15"
PROSPECTIVE_END_DATE = "2021-01-10"

# Ground truth sources.
GT_SOURCE_JHU = "JHU"
GT_SOURCE_NYT = "NYT"
GT_SOURCE_USAFACTS = "USAFACTS"
GT_SOURCE_JAPAN = "JAPAN"
GT_SOURCE_LIST = [
    GT_SOURCE_JHU, GT_SOURCE_NYT, GT_SOURCE_USAFACTS, GT_SOURCE_JAPAN
]

# Location granularity constants.
LOCATION_GRANULARITY_STATE = "STATE"
LOCATION_GRANULARITY_COUNTY = "COUNTY"
LOCATION_GRANULARITY_COUNTRY = "COUNTRY"
LOCATION_GRANULARITY_JAPAN_PREFECTURE = "JAPAN_PREFECTURE"
LOCATION_GRANULARITY_LIST = [
    LOCATION_GRANULARITY_STATE, LOCATION_GRANULARITY_COUNTY,
    LOCATION_GRANULARITY_COUNTRY, LOCATION_GRANULARITY_JAPAN_PREFECTURE
]

# Model types
MODEL_TYPE_STATIC_SEIR = "STATIC_SEIR"
MODEL_TYPE_TIME_VARYING_WITH_COVARIATES = "TIME_VARYING_WITH_COVARIATES"
MODEL_TYPE_TREND_FOLLOWING = "TREND_FOLLOWING"

# Ground-truth features
# We distinguish the names that occur in Bigquery table vs. how models are used.
# features.py provides mappings from feature keys to predicted metrics
JHU_CONFIRMED_FEATURE_KEY = "jhu_state_confirmed_cases"
JHU_COUNTY_CONFIRMED_FEATURE_KEY = "jhu_county_confirmed_cases"
NYT_CONFIRMED_FEATURE_KEY = "nyt_state_confirmed_cases"
NYT_COUNTY_CONFIRMED_FEATURE_KEY = "nyt_county_confirmed_cases"
USAFACTS_CONFIRMED_FEATURE_KEY = "confirmed_cases"  # USAFACTS
USAFACTS_COUNTY_CONFIRMED_FEATURE_KEY = "confirmed_cases"  # USAFACTS
CONFIRMED_FEATURE_KEY = "confirmed_cases"  # from JHU for countries
JHU_DEATH_FEATURE_KEY = "jhu_state_deaths"
JHU_COUNTY_DEATH_FEATURE_KEY = "jhu_county_deaths"
NYT_COUNTY_DEATH_FEATURE_KEY = "nyt_county_deaths"
NYT_DEATH_FEATURE_KEY = "nyt_state_deaths"
USAFACTS_DEATH_FEATURE_KEY = "deaths"  # from USAFACTS for US locations
DEATH_FEATURE_KEY = "deaths"  # from JHU for countries
INFECTED_FEATURE_KEY = "infected_cases"
CTP_ICU_FEATURE_KEY = "inIcuCurrently"
CTP_HOSPITALIZED_FEATURE_KEY = "hospitalizedCurrently"
CTP_HOSPITALIZED_CUMULATIVE_FEATURE_KEY = "hospitalizedCumulative"
CTP_HOSPITALIZED_INCREASE_FEATURE_KEY = "hospitalizedIncrease"
VENTILATOR_FEATURE_KEY = "onVentilatorCurrently"
RECOVERED_FEATURE_KEY = "recovered"  # from Covidtracking for states
CSRP_HOSPITALIZED_FEATURE_KEY = "coronadatascraper_hospitalized_current"
CSRP_HOSPITALIZED_CUMULATIVE_FEATURE_KEY = "coronadatascraper_hospitalized"
CSRP_ICU_FEATURE_KEY = "coronadatascraper_icu_current"
CSRP_RECOVERED_FEATURE_KEY = "coronadatascraper_recovered"

# CHA hospitalized metrics to update CSRP metrics listed above
# Replaces coronadatascraper_hospitalized_current
CHA_HOSPITALIZED_FEATURE_KEY = "cha_confirmed_patients"
# Replaces coronadatascraper_hospitalized
CHA_HOSPITALIZED_CUMULATIVE_FEATURE_KEY = "cha_total_admits_confirmed"

# HHS Hospitalization data features (US state-level)
HHS_HOSPITALIZED_FEATURE_KEY = "hhs_hosp_total_patients_hospitalized_confirmed_and_suspected_covid"
HHS_HOSPITALIZED_INCREASE_FEATURE_KEY = "hhs_hosp_previous_day_admission_total_covid_suspected_confirmed"
HHS_ICU_FEATURE_KEY = "hhs_hosp_staffed_icu_adult_patients_confirmed_and_suspected_covid"

# Selected US state-level hospitalization ground truth features from the
# available choices of source:
# CovidTracking / HHS
# - these are required in every model version
HOSPITALIZED_FEATURE_KEY = HHS_HOSPITALIZED_FEATURE_KEY
HOSPITALIZED_INCREASE_FEATURE_KEY = HHS_HOSPITALIZED_INCREASE_FEATURE_KEY
ICU_FEATURE_KEY = HHS_ICU_FEATURE_KEY
# Cumulative hospitalized and icu have no equivalent so we compute it as an
# engineered feature.
HHS_HOSPITALIZED_CUMULATIVE_FEATURE_KEY = "hhs_hosp_hospitalized_cumulative"
HOSPITALIZED_CUMULATIVE_FEATURE_KEY = HHS_HOSPITALIZED_CUMULATIVE_FEATURE_KEY
HHS_ICU_CUMULATIVE_FEATURE_KEY = "hhs_hosp_icu_cumulative"
ICU_CUMULATIVE_FEATURE_KEY = HHS_ICU_CUMULATIVE_FEATURE_KEY

# Metrics to be predicted
CONFIRMED = "confirmed"
INFECTED = "infected"
INFECTED_DOC = "infected_documented"
INFECTED_UNDOC = "infected_undocumented"
INFECTED_UNDOC_INCREASE = "infected_undocumented_increase"
RECOVERED = "recovered"
RECOVERED_DOC = "recovered_documented"
RECOVERED_UNDOC = "recovered_undocumented"
DEATH = "death"
INCREMENTAL_DEATH = "death_daily_incremental"
INCREMENTAL_CONFIRMED = "confirmed_daily_incremental"
HORIZON_AHEAD_DEATH = "death_horizon_ahead"
HORIZON_AHEAD_CONFIRMED = "confirmed_horizon_ahead"

# NOTE: HOSPITALIZED is the number of people *in* the hospital each day.
# HOSPITALIZED_INCREASE is the number of people who *enter* the hospital each
# day. In particular, HOSPITALIZED, but not HOSPITALIZED_INCREASE, accounts for
# people leaving the hospital due to recovery or death; HOSPITALIZED_INCREASE is
# always non-negative and is *not* the daily increment in HOSPITALIZED.
HOSPITALIZED = "hospitalized"
HOSPITALIZED_INCREASE = "hospitalized_increase"
HOSPITALIZED_CUMULATIVE = "hospitalized_cumulative"
ICU = "icu"
VENTILATOR = "ventilator"
SUSCEPTIBLE = "susceptible"
EXPOSED = "exposed"

DEATH_PREPROCESSED = "death_preprocessed"
CONFIRMED_PREPROCESSED = "confirmed_preprocessed"
DEATH_PREPROCESSED_MEAN_TO_SUM_RATIO = "preprocessed_death_mean_to_sum"
CONFIRMED_PREPROCESSED_MEAN_TO_SUM_RATIO = "preprocessed_confirmed_mean_to_sum"

# Static features
DENSITY = "density"
POPULATION = "population"
INCOME_PER_CAPITA = "income_per_capita"
POPULATION_DENSITY = "population_density"
AREA = "area_land_meters"
HOUSEHOLDS = "households"
HOUSEHOLD_FOOD_STAMP = "households_public_asst_or_food_stamps"
KAISER_POPULATION = "kaiser_population"
PREPROCESSED_POPULATION = "preprocessed_population"
KAISER_60P_POPULATION = "kaiser_60plus_population"
POPULATION_DENSITY_PER_SQKM = "population_density_per_sq_km"
POPULATION_60P_RATIO = "population_60plus_ratio"

# mask-usage static features from NYT - applies to both COUNTY and STATE models.
# STATE-model features are weighted by county populations.
NYT_MASK_USE_NEVER = "nyt_mask_use_never"
NYT_MASK_USE_RARELY = "nyt_mask_use_rarely"
NYT_MASK_USE_SOMETIMES = "nyt_mask_use_sometimes"
NYT_MASK_USE_FREQUENTLY = "nyt_mask_use_frequently"
NYT_MASK_USE_ALWAYS = "nyt_mask_use_always"

# static features pertaining to county-level models
COUNTY_POPULATION = "census_2018_5yr_total_pop"
COUNTY_POVERTY = "census_2018_5yr_poverty"
COUNTY_POP_BASED_POVERTY = "census_2018_5yr_pop_determined_poverty_status"
COUNTY_MEDIAN_INCOME = "census_2018_5yr_median_income"
COUNTY_HOUSEHOLD_FOOD_STAMP = "census_2018_5yr_households_public_asst_or_food_stamps"
COUNTY_INCOME_PER_CAPITA = "census_2018_5yr_income_per_capita"
COUNTY_HOUSEHOLDS = "census_2018_5yr_households"
COUNTY_GROUP_QUARTERS = "census_2018_5yr_group_quarters"

# county static features for race distribution excluding 0.1-1.0% undeclared.
COUNTY_RACE_AMERINDIAN = "census_2018_5y_amerindian_pop"
COUNTY_RACE_ASIAN_POP = "census_2018_5y_asian_pop"
COUNTY_RACE_BLACK_POP = "census_2018_5y_black_pop"
COUNTY_RACE_HISPANIC_POP = "census_2018_5y_hispanic_pop"
COUNTY_RACE_OTHER_RACE_POP = "census_2018_5y_other_race_pop"
COUNTY_RACE_TWO_OR_MORE_RACES_POP = "census_2018_5y_two_or_more_races_pop"
COUNTY_RACE_WHITE_POP = "census_2018_5y_white_pop"
# sum of above populations, usually less than total_pop.
COUNTY_RACE_DECLARED_POP = "census_2018_5y_race_declared_pop"


# County static features for age distribution
# pylint: disable=line-too-long
# Ranges balance between # features and age resolution. Other sources:
#  https://www.cdc.gov/coronavirus/2019-ncov/covid-data/investigations-discovery/hospitalization-death-by-age.html
#  https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Sex-Age-and-S/9bhg-hcku
#. https://www.cdc.gov/vaccines/acip/meetings/downloads/slides-2020-08/COVID-06-Slayton.pdf
# pylint: enable=line-too-long
COUNTY_POP_AGE_RANGES = {
    "census_population_fraction_00_to_17": [
        "census_2018_5yr_age_00_to_04_pop",
        "census_2018_5yr_age_05_to_09_pop",
        "census_2018_5yr_age_10_to_14_pop",
        "census_2018_5yr_age_15_to_17_pop",
    ],
    "census_population_fraction_18_to_64": [
        "census_2018_5yr_age_18_to_19_pop",
        "census_2018_5yr_age_20_to_20_pop",
        "census_2018_5yr_age_21_to_21_pop",
        "census_2018_5yr_age_22_to_24_pop",
        "census_2018_5yr_age_25_to_29_pop",
        "census_2018_5yr_age_30_to_34_pop",
        "census_2018_5yr_age_35_to_39_pop",
        "census_2018_5yr_age_40_to_44_pop",
        "census_2018_5yr_age_45_to_49_pop",
        "census_2018_5yr_age_50_to_54_pop",
        "census_2018_5yr_age_55_to_59_pop",
        "census_2017_5yr_age_60_to_61_pop",
        "census_2017_5yr_age_62_to_64_pop",
    ],
    "census_population_fraction_65_to_99": [
        "census_2018_5yr_age_65_to_66_pop",
        "census_2018_5yr_age_67_to_69_pop",
        "census_2018_5yr_age_70_to_74_pop",
        "census_2018_5yr_age_75_to_79_pop",
        "census_2018_5yr_age_80_to_84_pop",
        "census_2018_5yr_age_85_to_99_pop",
    ],
}

# other static features
PATIENCE_EXPERIENCE_ABOVE = "patient_experience_above_the_national_average"
PATIENCE_EXPERIENCE_BELOW = "patient_experience_below_the_national_average"
PATIENCE_EXPERIENCE_SAME = "patient_experience_same_as_the_national_average"
EOC_CARE_ABOVE = "eoc_above_the_national_average"
EOC_CARE_BELOW = "eoc_below_the_national_average"
EOC_CARE_SAME = "eoc_same_as_the_national_average"
CRITICAL_ACCESS_HOSPITAL = "hospital_type_critical_access_hospitals"
HOSPITAL_ACUTE_CARE = "hospital_type_acute_care_hospitals"
EMERGENCY_SERVICES = "service_type_emergency_services_supported"
NON_EMERGENCY_SERVICES = "service_type_non_emergency_services"
AQI_MEAN = "aqi_mean_2018"
HOSPITAL_RATING1 = "rating_1"
HOSPITAL_RATING2 = "rating_2"
HOSPITAL_RATING3 = "rating_3"
HOSPITAL_RATING4 = "rating_4"
HOSPITAL_RATING5 = "rating_5"
HOSPITAL_RATING_AVERAGE = "rating_average"
ICU_BEDS = "icu_beds"

# Geolocal static covariates
ICU_BEDS_MEAN = "icu_beds_mean"
ICU_BEDS_MEDIAN = "icu_beds_median"
ICU_BEDS_STD = "icu_beds_std"
ICU_BEDS_MAX = "icu_beds_max"
ICU_BEDS_SUM = "icu_beds_sum"
HOSPITAL_ACUTE_CARE_MEAN = "hospital_type_acute_care_hospitals_mean"
HOSPITAL_ACUTE_CARE_MEDIAN = "hospital_type_acute_care_hospitals_median"
HOSPITAL_ACUTE_CARE_STD = "hospital_type_acute_care_hospitals_std"
HOSPITAL_ACUTE_CARE_MAX = "hospital_type_acute_care_hospitals_max"
HOSPITAL_ACUTE_CARE_SUM = "hospital_type_acute_care_hospitals_sum"

# Time-varying covariates
MOBILITY = "m50"
MOBILITY_INDEX = "m50_index"
MOBILITY_SAMPLES = "mobility_samples"
GOOGLE_MOBILITY_PARKS = "google_mobility_parks_percent_change_from_baseline"
GOOGLE_MOBILITY_WORK = "google_mobility_workplaces_percent_change_from_baseline"
GOOGLE_MOBILITY_RES = "google_mobility_residential_percent_change_from_baseline"
GOOGLE_MOBILITY_TRANSIT = "google_mobility_transit_stations_percent_change_from_baseline"
GOOGLE_MOBILITY_GROCERY = "google_mobility_grocery_and_pharmacy_percent_change_from_baseline"
GOOGLE_MOBILITY_RETAIL = "google_mobility_retail_and_recreation_percent_change_from_baseline"
GOOGLE_MOBILITY_FEATURES = [
    GOOGLE_MOBILITY_RETAIL, GOOGLE_MOBILITY_GROCERY, GOOGLE_MOBILITY_TRANSIT,
    GOOGLE_MOBILITY_RES, GOOGLE_MOBILITY_WORK, GOOGLE_MOBILITY_PARKS
]
TOTAL_TESTS = "totalTestResults"
CONFIRMED_PER_TESTS = "confirmed_per_totalTestResults"
TOTAL_TESTS_PER_CAPITA = "totalTestPerCapita"
CSRP_TESTS = "coronadatascraper_tested"
CONFIRMED_PER_CSRP_TESTS = "coronadatascraper_confirmed_per_tested"
AVERAGE_TEMPERATURE = "open_weather_average_temperature_celsius"
MAX_TEMPERATURE = "open_weather_maximum_temperature_celsius"
MIN_TEMPERATURE = "open_weather_minimum_temperature_celsius"
RAINFALL = "open_weather_rainfall_mm"
SNOWFALL = "open_weather_snowfall_mm"
COMMERCIAL_SCORE = "commercialScore"
ANTIGEN_POSITIVE = "positiveTestsAntigen"
ANTIGEN_TOTAL = "totalTestsPeopleAntigen"
ANTIGEN_POSITIVE_RATIO = "positiveRatioAntigen"
ANTIBODY_NEGATIVE = "negativeTestsAntibody"
ANTIBODY_TOTAL = "totalTestsAntibody"
ANTIBODY_NEGATIVE_RATIO = "negativeRatioAntibody"
FB_MOVEMENT_CHANGE = "movement_relative_change"
FB_MOVEMENT_STAYING_PUT = "fraction_staying_put"
SYMPTOM_COUGH = "symptom_Cough"
SYMPTOM_CHILLS = "symptom_Chills"
SYMPTOM_ANOSMIA = "symptom_Anosmia"
SYMPTOM_INFECTION = "symptom_Infection"
SYMPTOM_CHEST_PAIN = "symptom_Chest_pain"
SYMPTOM_FEVER = "symptom_Fever"
SYMPTOM_SHORTNESSBREATH = "symptom_Shortness_of_breath"
ALL_SYMPTOMS = (SYMPTOM_COUGH, SYMPTOM_CHILLS, SYMPTOM_ANOSMIA,
                SYMPTOM_INFECTION, SYMPTOM_CHEST_PAIN, SYMPTOM_FEVER,
                SYMPTOM_SHORTNESSBREATH)

# Day of week based covariates
DAY_OF_WEEK = "day_of_week"
WEEKEND_DAY = "weekend_day"
MONDAY = "monday"
TUESDAY = "tuesday"
WEDNESDAY = "wednesday"
THURSDAY = "thursday"
FRIDAY = "friday"
SATURDAY = "saturday"
SUNDAY = "sunday"
DOW_WINDOW = "dow_window"
DAY_OF_WEEK_FEATURES = frozenset((
    DAY_OF_WEEK,
    WEEKEND_DAY,
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    FRIDAY,
    SATURDAY,
    SUNDAY,
    DOW_WINDOW,
))

# Geolocal time-varying covariates
MOBILITY_MEAN = "m50_mean"
MOBILITY_MEDIAN = "m50_median"
MOBILITY_STD = "m50_std"
MOBILITY_MAX = "m50_max"
MOBILITY_SUM = "m50_sum"
MOBILITY_INDEX_MEAN = "m50_index_mean"
MOBILITY_INDEX_MEDIAN = "m50_index_median"
MOBILITY_INDEX_STD = "m50_index_std"
MOBILITY_INDEX_MAX = "m50_index_max"
MOBILITY_INDEX_SUM = "m50_index_sum"
MOBILITY_SAMPLES_MEAN = "mobility_samples_mean"
MOBILITY_SAMPLES_MEDIAN = "mobility_samples_median"
MOBILITY_SAMPLES_STD = "mobility_samples_std"
MOBILITY_SAMPLES_MAX = "mobility_samples_max"
MOBILITY_SAMPLES_SUM = "mobility_samples_sum"
USAFACTS_CONFIRMED_CASES_MEAN = "confirmed_cases_mean"
USAFACTS_CONFIRMED_CASES_MEDIAN = "confirmed_cases_median"
USAFACTS_CONFIRMED_CASES_STD = "confirmed_cases_std"
USAFACTS_CONFIRMED_CASES_MAX = "confirmed_cases_max"
USAFACTS_CONFIRMED_CASES_SUM = "confirmed_cases_sum"
USAFACTS_DEATHS_MEAN = "deaths_mean"
USAFACTS_DEATHS_MEDIAN = "deaths_median"
USAFACTS_DEATHS_STD = "deaths_std"
USAFACTS_DEATHS_MAX = "deaths_max"
USAFACTS_DEATHS_SUM = "deaths_sum"
NYT_COUNTY_CONFIRMED_CASES_MEAN = "nyt_county_confirmed_cases_mean"
NYT_COUNTY_CONFIRMED_CASES_MEDIAN = "nyt_county_confirmed_cases_median"
NYT_COUNTY_CONFIRMED_CASES_STD = "nyt_county_confirmed_cases_std"
NYT_COUNTY_CONFIRMED_CASES_MAX = "nyt_county_confirmed_cases_max"
NYT_COUNTY_CONFIRMED_CASES_SUM = "nyt_county_confirmed_cases_sum"
NYT_COUNTY_DEATHS_MEAN = "nyt_county_deaths_mean"
NYT_COUNTY_DEATHS_MEDIAN = "nyt_county_deaths_median"
NYT_COUNTY_DEATHS_STD = "nyt_county_deaths_std"
NYT_COUNTY_DEATHS_MAX = "nyt_county_deaths_max"
NYT_COUNTY_DEATHS_SUM = "nyt_county_deaths_sum"
JHU_COUNTY_CONFIRMED_CASES_MEAN = "jhu_county_confirmed_cases_mean"
JHU_COUNTY_CONFIRMED_CASES_MEDIAN = "jhu_county_confirmed_cases_median"
JHU_COUNTY_CONFIRMED_CASES_STD = "jhu_county_confirmed_cases_std"
JHU_COUNTY_CONFIRMED_CASES_MAX = "jhu_county_confirmed_cases_max"
JHU_COUNTY_CONFIRMED_CASES_SUM = "jhu_county_confirmed_cases_sum"
JHU_COUNTY_DEATHS_MEAN = "jhu_county_deaths_mean"
JHU_COUNTY_DEATHS_MEDIAN = "jhu_county_deaths_median"
JHU_COUNTY_DEATHS_STD = "jhu_county_deaths_std"
JHU_COUNTY_DEATHS_MAX = "jhu_county_deaths_max"
JHU_COUNTY_DEATHS_SUM = "jhu_county_deaths_sum"
CSRP_TESTS_MEAN = "coronadatascraper_tested_mean"
CSRP_TESTS_MEDIAN = "coronadatascraper_tested_median"
CSRP_TESTS_STD = "coronadatascraper_tested_std"
CSRP_TESTS_MAX = "coronadatascraper_tested_max"
CSRP_TESTS_SUM = "coronadatascraper_tested_sum"

# Vaccine constants
VACCINE_IMMUNITY_DURATION = 180.0
VACCINE_EFFECTIVENESS_CHANGE_PERIOD = 14.0
FIRST_DOSE_VACCINE_MAX_EFFECT = 0.921
SECOND_DOSE_VACCINE_MAX_EFFECT = 0.945

# Govex (https://github.com/govex/COVID-19/tree/master/data_tables/vaccine_data)
VACCINES_GOVEX_FIRST_DOSE_TOTAL = "govex_vaccines_doses_admin_all"
VACCINES_GOVEX_SECOND_DOSE_TOTAL = "govex_vaccines_stage_two_doses_all"

# Japan MHLW vaccination data from
# https://github.com/swsoyee/2019-ncov-japan/tree/master/50_Data/MHLW
VACCINES_JAPAN_FIRST_DOSE = "covidlive_japan_vaccine_doses_first"
VACCINES_JAPAN_SECOND_DOSE = "covidlive_japan_vaccine_doses_second"
VACCINES_JAPAN_TOTAL_DOSE = "covidlive_japan_vaccine_doses_total"

# derived vaccine distribution features
VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED = "vaccinated_ratio_first_dose_per_day_preprocessed"
VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED = "vaccinated_ratio_second_dose_per_day_preprocessed"

# derived vaccine effectiveness features
VACCINATED_EFFECTIVENESS_FIRST_DOSE = "vaccinated_effectiveness_first_dose"
VACCINATED_EFFECTIVENESS_SECOND_DOSE = "vaccinated_effectiveness_second_dose"

VACCINATED_RATIO = "vaccinated_ratio"
VACCINE_EFFECTIVENESS = "vaccine_effectiveness"

STATE_VACCINE_FEATURES = [
    VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED,
    VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED,
    VACCINATED_EFFECTIVENESS_FIRST_DOSE,
    VACCINATED_EFFECTIVENESS_SECOND_DOSE,
]
# US county- and Japan prefecture-level vaccine feature
VACCINE_FEATURES = [
    VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED,
    VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED,
    VACCINATED_EFFECTIVENESS_FIRST_DOSE,
    VACCINATED_EFFECTIVENESS_SECOND_DOSE,
]

# Japan-specific features
# Ground truth from the Kaz-Ogiwara dataset
# (https://github.com/kaz-ogiwara/covid19/tree/master/data)
JAPAN_PREFECTURE_KAZ_DEATH_FEATURE_KEY = "kaz_deaths"
JAPAN_PREFECTURE_KAZ_CONFIRMED_FEATURE_KEY = "kaz_confirmed_cases"
# Ground truth from Covid Open data (2nd source)
# 'recovered' is not included because all values are null.
JAPAN_PREFECTURE_OPEN_DEATH_FEATURE_KEY = "open_gt_jp_deaths"
JAPAN_PREFECTURE_OPEN_CONFIRMED_FEATURE_KEY = "open_gt_jp_confirmed_cases"
JAPAN_PREFECTURE_OPEN_TESTED_FEATURE_KEY = "open_gt_jp_num_tested"
JAPAN_PREFECTURE_OPEN_HOSPITALIZED_FEATURE_KEY = "open_gt_jp_hospitalized_cases"
JAPAN_PREFECTURE_OPEN_ICU_FEATURE_KEY = "open_gt_jp_icu_cases"
JAPAN_PREFECTURE_OPEN_VENTILATOR_FEATURE_KEY = "open_gt_jp_ventilator_cases"
# Ground truth from Covid.Live data (3rd source).
# (https://github.com/swsoyee/2019-ncov-japan)
# This source also includes hospitalizations, which is no longer available with
# Kaz.
JAPAN_PREFECTURE_COVIDLIVE_DEATH_FEATURE_KEY = "covidlive_gt_jp_deaths"
JAPAN_PREFECTURE_COVIDLIVE_CONFIRMED_FEATURE_KEY = "covidlive_gt_jp_confirmed_cases"
JAPAN_PREFECTURE_COVIDLIVE_TESTED_FEATURE_KEY = "covidlive_gt_jp_num_tested"
JAPAN_PREFECTURE_COVIDLIVE_DISCHARGED_FEATURE_KEY = "covidlive_gt_jp_discharged"
JAPAN_PREFECTURE_COVIDLIVE_HOSPITALIZED_FEATURE_KEY = "covidlive_gt_jp_hospitalized"
JAPAN_PREFECTURE_COVIDLIVE_SERIOUS_FEATURE_KEY = "covidlive_gt_jp_serious"

# Engineered features for Japan
JAPAN_PREFECTURE_DEATH_PREPROCESSED_MEAN_TO_SUM_RATIO = "death_mean_to_sum_ratio"
JAPAN_PREFECTURE_CONFIRMED_PREPROCESSED_MEAN_TO_SUM_RATIO = "confirmed_mean_to_sum_ratio"

# Static features (Japan).
# Available hospital resources.
JAPAN_PREFECTURE_NUM_DOCTORS_FEATURE_KEY = "doctors"
JAPAN_PREFECTURE_DOCTORS_PER_100K_FEATURE_KEY = "doctors_per_100k"
JAPAN_PREFECTURE_NUM_HOSPITAL_BEDS_FEATURE_KEY = "hospital_beds"
JAPAN_PREFECTURE_NUM_HOSPITAL_BEDS_PER_100K_FEATURE_KEY = "hospital_beds_per_100k"
JAPAN_PREFECTURE_NUM_CLINIC_BEDS_FEATURE_KEY = "clinic_beds"
JAPAN_PREFECTURE_NUM_CLINIC_BEDS_PER_100K_FEATURE_KEY = "clinic_beds_per_100k"
JAPAN_PREFECTURE_NUM_NEW_ICU_BEDS_FEATURE_KEY = "new_beds"
# Population and demographics.
JAPAN_PREFECTURE_NUM_PEOPLE_FEATURE_KEY = "population_total"
JAPAN_PREFECTURE_NUM_MALE_FEATURE_KEY = "population_male"
JAPAN_PREFECTURE_NUM_FEMALE_FEATURE_KEY = "population_female"
JAPAN_PREFECTURE_POPULATION_DENSITY_FEATURE_KEY = "population_density"
JAPAN_PREFECTURE_AGE_0_TO_14_FEATURE_KEY = "age_0_to_14"
JAPAN_PREFECTURE_AGE_15_TO_64_FEATURE_KEY = "age_15_to_64"
JAPAN_PREFECTURE_AGE_64_PLUS_FEATURE_KEY = "age_64_and_over"
JAPAN_PREFECTURE_AGE_75_PLUS_FEATURE_KEY = "age_75_and_over"
JAPAN_PREFECTURE_GDP_PER_CAPITA_FEATURE_KEY = "gdp_per_capita"
# Wellness and health.
JAPAN_PREFECTURE_H1N1_in_2010_FEATURE_KEY = "h1n1_in_2010"  # pylint:disable=invalid-name
JAPAN_PREFECTURE_ALCOHOL_INTAKE_SCORE_FEATURE_KEY = "alcohol_intake"
JAPAN_PREFECTURE_BMI_MALE_AVERAGE_FEATURE_KEY = "bmi_male_average"
JAPAN_PREFECTURE_BMI_MALE_LOWER_RANGE_FEATURE_KEY = "bmi_male_confidence_interval_lower"
JAPAN_PREFECTURE_BMI_MALE_UPPER_RANGE_FEATURE_KEY = "bmi_male_confidence_interval_upper"
JAPAN_PREFECTURE_BMI_FEMALE_AVERAGE_FEATURE_KEY = "bmi_female_average"
JAPAN_PREFECTURE_BMI_FEMALE_LOWER_RANGE_FEATURE_KEY = "bmi_female_confidence_interval_lower"
JAPAN_PREFECTURE_BMI_FEMALE_UPPER_RANGE_FEATURE_KEY = "bmi_female_confidence_interval_upper"
JAPAN_PREFECTURE_SMOKERS_MALE_FEATURE_KEY = "perfect_smoker_male"
JAPAN_PREFECTURE_SMOKERS_FEMALE_FEATURE_KEY = "perfect_smoker_female"

# Timeseries features (Japan).
# These features names are constructed and exported here:
# ../etl/R/jp_*.R
JAPAN_PREFECTURE_STATE_OF_EMERGENCY_FEATURE_KEY = "soe_soe"
JAPAN_PREFECTURE_HOSPITAL_DISCHARGED_FEATURE_KEY = "kaz_discharged"
JAPAN_PREFECTURE_NUMBER_TESTED_FEATURE_KEY = "kaz_num_tested"
JAPAN_PREFECTURE_NUMBER_HOSPITALIZED_FEATURE_KEY = "kaz_hospitalized"
JAPAN_PREFECTURE_NUMBER_SERIOUS_FEATURE_KEY = "kaz_serious"
JAPAN_PREFECTURE_EFFECTIVE_REPRODUCTIVE_NUMBER_FEATURE_KEY = "kaz_effective_reproductive_number"
JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_SURVEY_FEATURE_KEY = "mar_cli_se"
JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_UNWEIGHTED_SURVEY_FEATURE_KEY = "mar_cli_se_unw"
JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_PERCENT_SURVEY_FEATURE_KEY = "mar_percent_cli"
JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_PERCENT_UNWEIGHTED_SURVEY_FEATURE_KEY = "mar_percent_cli_unw"
JAPAN_PREFECTURE_SURVEY_SAMPLE_SIZE_FEATURE_KEY = "mar_sample_size"
JAPAN_PREFECTURE_MOBILITY_PARKS_PERCENT_FROM_BASELINE_FEATURE_KEY = "google_mobility_parks_percent_change_from_baseline"
JAPAN_PREFECTURE_MOBILITY_WORKPLACE_PERCENT_FROM_BASELINE_FEATURE_KEY = "google_mobility_workplaces_percent_change_from_baseline"
JAPAN_PREFECTURE_MOBILITY_RESIDENTIAL_PERCENT_FROM_BASELINE_FEATURE_KEY = "google_mobility_residential_percent_change_from_baseline"
JAPAN_PREFECTURE_MOBILITY_TRAIN_STATION_PERCENT_FROM_BASELINE_FEATURE_KEY = "google_mobility_transit_stations_percent_change_from_baseline"
JAPAN_PREFECTURE_MOBILITY_GROCERY_AND_PHARMACY_PERCENT_FROM_BASELINE_FEATURE_KEY = "google_mobility_grocery_and_pharmacy_percent_change_from_baseline"
JAPAN_PREFECTURE_MOBILITY_RETAIL_AND_RECREATION_PERCENT_FROM_BASELINE_FEATURE_KEY = "google_mobility_retail_and_recreation_percent_change_from_baseline"

# Selected ground truth features from the available choices of source:
# Kaz / Open Covid / Covid.Live
# - these are required in every model version
JAPAN_PREFECTURE_DEATH_FEATURE_KEY = JAPAN_PREFECTURE_OPEN_DEATH_FEATURE_KEY
JAPAN_PREFECTURE_CONFIRMED_FEATURE_KEY = JAPAN_PREFECTURE_OPEN_CONFIRMED_FEATURE_KEY
JAPAN_PREFECTURE_HOSPITALIZED_FEATURE_KEY = JAPAN_PREFECTURE_COVIDLIVE_HOSPITALIZED_FEATURE_KEY
JAPAN_PREFECTURE_DISCHARGED_FEATURE_KEY = JAPAN_PREFECTURE_COVIDLIVE_DISCHARGED_FEATURE_KEY
JAPAN_PREFECTURE_TESTED_FEATURE_KEY = JAPAN_PREFECTURE_COVIDLIVE_TESTED_FEATURE_KEY

# Japan Graph Model: graph features for Japan are inserted dynamically, so do
# not add them here.

# CHC intervention features, binary indicators denoting the bans on social life
CHC_REST = "chc_npi_Bar_Rest"
CHC_BUS = "chc_npi_NE_Bus"
CHC_MOVE = "chc_npi_Move"
CHC_SCHOOL = "chc_npi_School"
CHC_EMER = "chc_npi_EMER"
CHC_GATHER = "chc_npi_Gatherings"
CHC_MASK = "chc_npi_Masks"

# AMP intervention features
AMP_GATHERINGS = "social_distancing__mass_gathering_restrictions__general_population_inclusive"
AMP_RESTAURANTS = "social_distancing__private_sector_closures__restaurants_bars"
AMP_CONCERTS = "social_distancing__private_sector_closures__entertainment_concert_venues_nightclubs"
AMP_FITNESS = "social_distancing__private_sector_closures__gym_workout_facility_fitness_center"
AMP_PERSONAL_CARE = "social_distancing__private_sector_closures__personal_care_facilities"
AMP_CASINOS = "social_distancing__private_sector_closures__casinos_gambling_facilities"
AMP_PRIVATE_PRIMARY_EDUCATION = "social_distancing__private_sector_closures__primary_education"
AMP_PRIVATE_CHILDCARE = "social_distancing__private_sector_closures__childcare_facilities_preschool_or_other"
AMP_NON_ESSENTIAL_BUSINESS = "social_distancing__private_sector_closures__non_essential_business_inclusive"
AMP_PRIVATE_INDOOR_RECREATION = "social_distancing__private_sector_closures__indoor_recreation_sports_facility"
AMP_PRIVATE_OUTDOOR_RECREATION = "social_distancing__private_sector_closures__outdoor_recreation_or_space"
AMP_STAY_AT_HOME = "social_distancing__stay_at_home__general_population_inclusive"
AMP_PRIVATE_GENERAL_POPULATION = "social_distancing__private_sector_closures__general_population_inclusive"
AMP_CHURCHES = "social_distancing__private_sector_closures__churches_places_of_worship"
AMP_RETAIL = "social_distancing__private_sector_closures__retail_stores_not_including_food"
AMP_SCHOOLS_PRIMARY_EDUCATION = "social_distancing__school_closures__primary_education"
AMP_SCHOOLS_SECONDARY_EDUCATION = "social_distancing__school_closures__secondary_education"
AMP_SCHOOLS_HIGHER_EDUCATION = "social_distancing__school_closures__higher_education"
AMP_SCHOOLS = "social_distancing__school_closures__general_population_inclusive"
AMP_FACE_MASKS = "face_mask__face_mask_required__general_population_inclusive"
AMP_EMERGENCY_DECLARATION = "emergency_declarations__general_emergency_declaration__general_population_inclusive"
AMP_NON_ESSENTIAL_WORKERS = "social_distancing__private_sector_closures__non_essential_workers_inclusive"
AMP_CONTACT_TRACING = "contact_tracing_testing__contact_tracing__general_population_inclusive"
AMP_CURFEW = "social_distancing__curfews__general_population_inclusive"

# TODO(nyoder): Refactor this to be less directly spread across the code base
# list of categorical features
ALL_CATEGORICAL_FEATURES = frozenset(
    (CHC_REST, CHC_BUS, CHC_MOVE, CHC_SCHOOL, CHC_EMER, CHC_GATHER, CHC_MASK,
     JAPAN_PREFECTURE_STATE_OF_EMERGENCY_FEATURE_KEY, AMP_GATHERINGS,
     AMP_RESTAURANTS, AMP_CONCERTS, AMP_FITNESS, AMP_PERSONAL_CARE, AMP_CASINOS,
     AMP_PRIVATE_PRIMARY_EDUCATION, AMP_PRIVATE_CHILDCARE,
     AMP_NON_ESSENTIAL_BUSINESS, AMP_PRIVATE_INDOOR_RECREATION,
     AMP_PRIVATE_OUTDOOR_RECREATION, AMP_STAY_AT_HOME,
     AMP_PRIVATE_GENERAL_POPULATION, AMP_CHURCHES, AMP_RETAIL,
     AMP_SCHOOLS_PRIMARY_EDUCATION, AMP_SCHOOLS_SECONDARY_EDUCATION,
     AMP_SCHOOLS_HIGHER_EDUCATION, AMP_SCHOOLS, AMP_FACE_MASKS,
     AMP_EMERGENCY_DECLARATION, AMP_NON_ESSENTIAL_WORKERS, AMP_CONTACT_TRACING,
     AMP_CURFEW))

# Buckets to save models.
BUCKET_WHAT_IF = "what_if_models"
PROD_MODELS_GCS_BUCKET = "prod_models"
MODELS_GCS_BUCKET = "covid_models"

# Bucket to save plots
BUCKET_PLOTS = "covid_plots"

# Bucket to save KFP run data
BUCKET_KFP_RUN_DATA = "covid-related-data"

# Region to use for BigQuery
# Must support BigQuery and AI Platform Vizier
# https://cloud.google.com/bigquery/docs/locations#regional-locations
PROJECT_GCP_REGION_NAME = "us-central1"

# Tensorboard configuation
TENSORBOARD_BUCKET = "covid-tensorboard-data"
TENSORBOARD_LOCAL_DIR = "/tmp/logs/"
TENSORBOARD_OFF = "OFF"
TENSORBOARD_LOG_ITERATIONS = 20

# Rate name lists for dynamics models for US county and state.
# Setting location_granularity to "COUNTRY" will raise an error.
# TODO(sinharaj): add rate list for COUNTRY' granularity
# If you are changing any of the parameters below, please make sure to update
# the model code and parameters too.
COUNTRY_RATE_LIST = [  # all rate names end with '_rate' for ease-of-use
    "average_contact_id_rate", "average_contact_iud_rate", "reinfectable_rate",
    "alpha_rate", "diagnosis_rate", "recovery_id_rate", "recovery_iud_rate",
    "death_id_rate"
]
# LINT.IfChange
# all rate names end with '_rate' for downstream ease-of-use
HOSPITAL_RATE_LIST = [
    "first_dose_vaccine_ratio_per_day", "second_dose_vaccine_ratio_per_day",
    "average_contact_id_rate", "average_contact_iud_rate", "reinfectable_rate",
    "alpha_rate", "diagnosis_rate", "recovery_id_rate", "recovery_iud_rate",
    "recovery_h_rate", "hospitalization_rate", "death_id_rate", "death_h_rate"
]

ICU_AND_VENTILATOR_RATE_LIST = [
    "first_dose_vaccine_ratio_per_day", "second_dose_vaccine_ratio_per_day",
    "average_contact_id_rate", "average_contact_iud_rate", "reinfectable_rate",
    "alpha_rate", "diagnosis_rate", "recovery_id_rate", "recovery_iud_rate",
    "recovery_h_rate", "recovery_i_rate", "recovery_v_rate",
    "hospitalization_rate", "icu_rate", "ventilator_rate", "death_id_rate",
    "death_h_rate", "death_i_rate", "death_v_rate"
]

QUANTILE_LIST = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
]

QUANTILE_LIST_REDUCED = [
    0.025,
    0.5,
    0.975,
]

LOCATION_SPECIFIC_SANITY_CHECK_CONSTANTS = {
    LOCATION_GRANULARITY_STATE: {
        "forecast_length": 28,
        "forecast_relative_change_threshold": 0.06,
        "forecast_relative_change_min_val": 10,
        "prediction_exceeds_population_threshold": 1.05,
        "death_fraction_threshold": 0.02,
        "infected_undoc_threshold_denom": 3,
        "infected_doc_threshold_denom": 5,
        "confirmed_threshold_denom": 5,
        "infected_doc_lower_bound": 0.9,
        "infected_doc_upper_bound": 1.1,
        "death_lower_bound": 0.9,
        "death_upper_bound": 1.1,
        "max_growth_exponent": 1.1,
        "recovered_doc_lower_bound": 0.8,
        "recovered_doc_upper_bound": 1.2,
        "hospitalized_lower_bound": 0.8,
        "hospitalized_upper_bound": 1.2,
        "confirmed_lower_bound": 0.85,
        "confirmed_upper_bound": 1.15,
        "confirmed_growth_base": 1.35,
        "non_confirmed_growth_base": 1.30,
        "growth_check_min_point_prediction_value": 10,
        "bounds_check_min_gt_value": 10,
        "confirmed_growth_gt_threshold_high": 11,
        "confirmed_growth_pred_threshold_high": 100,
        "confirmed_growth_gt_threshold_low": 3,
        "confirmed_growth_pred_threshold_low": 20
    },
    LOCATION_GRANULARITY_COUNTY: {
        "forecast_length": 28,
        "forecast_relative_change_threshold": 0.06,
        "forecast_relative_change_min_val": 10,
        "prediction_exceeds_population_threshold": 1.05,
        "death_fraction_threshold": 0.02,
        "infected_undoc_threshold_denom": 3,
        "infected_doc_threshold_denom": 5,
        "confirmed_threshold_denom": 5,
        "infected_doc_lower_bound": 0.9,
        "infected_doc_upper_bound": 1.1,
        "death_lower_bound": 0.9,
        "death_upper_bound": 1.1,
        "max_growth_exponent": 1.1,
        "recovered_doc_lower_bound": 0.8,
        "recovered_doc_upper_bound": 1.2,
        "hospitalized_lower_bound": 0.8,
        "hospitalized_upper_bound": 1.2,
        "confirmed_lower_bound": 0.85,
        "confirmed_upper_bound": 1.15,
        "confirmed_growth_base": 1.35,
        "non_confirmed_growth_base": 1.30,
        "growth_check_min_point_prediction_value": 10,
        "bounds_check_min_gt_value": 10,
        "confirmed_growth_gt_threshold_high": 11,
        "confirmed_growth_pred_threshold_high": 100,
        "confirmed_growth_gt_threshold_low": 3,
        "confirmed_growth_pred_threshold_low": 20
    },
    LOCATION_GRANULARITY_JAPAN_PREFECTURE: {
        "forecast_length": 28,
        "forecast_relative_change_threshold": 0.06,
        "forecast_relative_change_min_val": 10,
        "prediction_exceeds_population_threshold": 1.05,
        "death_fraction_threshold": 0.02,
        "infected_undoc_threshold_denom": 3,
        "infected_doc_threshold_denom": 5,
        "confirmed_threshold_denom": 5,
        "infected_doc_lower_bound": 0.9,
        "infected_doc_upper_bound": 1.1,
        "death_lower_bound": 0.9,
        "death_upper_bound": 1.1,
        "max_growth_exponent": 1.1,
        "recovered_doc_lower_bound": 0.8,
        "recovered_doc_upper_bound": 1.2,
        "hospitalized_lower_bound": 0.8,
        "hospitalized_upper_bound": 1.2,
        "confirmed_lower_bound": 0.85,
        "confirmed_upper_bound": 1.15,
        "confirmed_growth_base": 1.15,
        "non_confirmed_growth_base": 1.10,
        "growth_check_min_point_prediction_value": 10,
        "bounds_check_min_gt_value": 10,
        "confirmed_growth_gt_threshold_high": 11,
        "confirmed_growth_pred_threshold_high": 100,
        "confirmed_growth_gt_threshold_low": 3,
        "confirmed_growth_pred_threshold_low": 20
    }
}

# ALERTING/MONITORING CONSTANTS
HP_TUNING_CONTROLLER_METRIC_NAME = "hp_trials"
HP_TUNING_CONTROLLER_JOB_NAME = "controller"
HP_TUNING_CONTORLLER_FAILED_TRIALS_METRIC_NAME = "hp_failed_trials"
EXTRACT_FORECAST_JOB_NAME = "extract_forecast"
EXTRACT_FORECAST_NOT_ENOUGH_TRIALS_METRIC_NAME = "not_enough_trials"
