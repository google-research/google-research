# Setup -------------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")

pacman::p_load("tidyverse", "tidyjson")


# Functions for Creating Keys ---------------------------------------------


# function to extract all scales from an admin session
extract_all_item_ids <- function(admin_session) {
  # Extracts a list of item_ids for all measures and subscales in the
  # admin_session object
  #
  # Args:
  #   admin_session: The admin_session object
  #
  # Returns:
  #   A nested list of item_ids for each measure and subscale

  item_ids <- list()

  # extract list of all measures in admin_session
  measures <- admin_session$..JSON[[1]]$measures

  # loop over each measure
  for (measure in names(measures)) {
    item_ids[[measure]] <- list()

    # loop over each subscale for the current measure
    for (
      subscale in names(admin_session$..JSON[[1]]$measures[[measure]]$scales)) {
      # extract item_ids for the current subscale
      item_ids[[measure]][[subscale]] <-
        measures[[measure]]$scales[[subscale]]$item_ids |>
        unlist()
    }
  }

  return(item_ids)
}

# function to extract all and reverse-keyed IDs from an admin session
extract_all_and_reversed_item_ids <- function(admin_session) {
  # Extracts a list of all and reverse-keyed item_ids for all measures and
  #   subscales inthe admin_session object.
  #
  # Args:
  #   admin_session: The admin_session object
  #
  # Returns:
  #   A nested list of reverse-keyed item_ids for each measure and subscale

  item_ids <- list()

  # extract list of all measures in admin_session
  measures <- admin_session$..JSON[[1]]$measures

  # loop over each measure
  for (measure in names(measures)) {
    item_ids[[measure]] <- list()

    # loop over each subscale for the current measure
    for (
      subscale in names(admin_session$..JSON[[1]]$measures[[measure]]$scales)) {
      # extract all item_ids for the current subscale
      item_ids[[measure]][[subscale]][["item_ids"]] <-
        measures[[measure]]$scales[[subscale]]$item_ids |>
        unlist()

      # extract reversed item_ids for the current subscale
      item_ids[[measure]][[subscale]][["reverse_keyed_item_ids"]] <-
        measures[[measure]]$scales[[subscale]]$reverse_keyed_item_ids |>
        unlist()
    }
  }

  return(item_ids)
}

# function to prepend "-" to reverse-keyed item IDs
key_item_ids <- function(item_ids, reverse_keyed_item_ids) {

  # inner function to key one item ID
  key_one_item_id <- function(item_id) {
    if (item_id %in% reverse_keyed_item_ids) {
      return(paste0("-", item_id))
    } else {
      return(item_id)
    }
  }

  keyed_item_ids <- lapply(item_ids, key_one_item_id) |> unlist()

  return(keyed_item_ids)
}

# function that reverse-keys all item IDs
create_keys <- function(all_nested_item_ids) {
  nested_key <- lapply(all_nested_item_ids, function(measure) {
    lapply(measure, function(subscale) {
      key_item_ids(subscale$item_ids, subscale$reverse_keyed_item_ids)
    })
  })
  return(nested_key)
}

# main function; converts admin_session to a nested key
admin_session_to_nested_key <- function(admin_session) {

  # extract all nested item_ids
  all_nested_item_ids <- extract_all_and_reversed_item_ids(admin_session)

  # add reverse-key syntax to these item_ids
  nested_key = create_keys(all_nested_item_ids)

  return(nested_key)
}



# Functions for Loading Data ----------------------------------------------


# Functions for Slicing Data ----------------------------------------------

data_for_model_id <- function(scored_session_df, model_id_str) {
  # Subsets a scored session by model_id.

  result <- scored_session_df |>
    filter(model_id == model_id_str) # nolint: object_usage_linter.

  return(result)
}


# Psychometric Functions --------------------------------------------------

subscale_reliability <- function(
    admin_session, df, measure, subscale, metric, min = 1, max = 5) {
  # Calls psych::scoreItems for a given subscale to retrieve Cronbach's Alpha.
  #
  # Uses min and max arguments to ensure proper reverse-keying.
  #
  # Returns:
  #   alpha: A float estimate of Cronbach's Alpha.

  # validate inputs
  metric <- match.arg(metric, c("alpha", "G6", "omega"))

  # get nested key
  nested_key <- admin_session_to_nested_key(admin_session)

  # get all item IDs
  all_item_ids <- extract_all_item_ids(admin_session)

  # get subscale item IDs
  subscale_item_ids <- all_item_ids[[measure]][[subscale]]

  # score subscale
  if (metric %in% c("alpha", "G6")) {
    score_info <- psych::scoreItems(
      keys = nested_key[[measure]][[subscale]],
      items = df |>
        dplyr::select(dplyr::all_of(all_item_ids[[measure]][[subscale]])),
      missing = FALSE,
      delete = TRUE,
      min = min,
      max = max
    )

    alpha <- score_info$alpha[1]
    G6 <- score_info$G6[1]

    if (metric == "alpha") {
      return(alpha)
    } else if (metric == "G6") {
      return(G6)
    }
  }

  if (metric == "omega") {
    omega_info <- psych::omega(
      df |> dplyr::select(dplyr::all_of(subscale_item_ids)) |>
        tidyr::drop_na(),
      plot = FALSE
    )
    omega <- omega_info |> purrr::pluck("omega.tot")

    return(omega)
  }
}

score_subscale <- function(
    admin_session, df, measure, subscale, min = 1, max = 5) {
  # Calls psych::scoreItems for a given subscale.
  #
  # Args:
  #   df: A DataFrame containing item-level response data.
  #   measure: String ID of measure.
  #   subscale: String ID of subscale.

  # get nested key
  nested_key = admin_session_to_nested_key(admin_session)

  # get all item IDs
  all_item_ids = extract_all_item_ids(admin_session)

  score_info <- psych::scoreItems(
    keys = nested_key[[measure]][[subscale]],
    items = df |> select(all_of(all_item_ids[[measure]][[subscale]])),
    missing = FALSE,
    delete = TRUE,
    min = min,
    max = max
  )

  alpha = score_info$alpha[1]
  G6 = score_info$G6[1]

  return(as.list(data.frame(alpha, G6)))
  # return(score_info$alpha |> pluck(1))
  # return(score_info)
}

score_measure <- function(admin_session, df, measure, min = 1, max = 5) {
  # Calls psych::scoreItems for a given subscale.
  #
  # Args:
  #   df: A DataFrame containing item-level response data.
  #   measure: String ID of measure.
  #   subscale: String ID of subscale.

  # get nested key
  nested_key = admin_session_to_nested_key(admin_session)

  # get all item IDs
  all_item_ids = extract_all_item_ids(admin_session)

  # measure IDs
  measure_item_ids = all_item_ids[[measure]] |>
    unlist(recursive = F) |> unname()

  score_info <- psych::scoreItems(
    keys = nested_key[[measure]],
    items = df |> select(all_of(measure_item_ids)),
    missing = FALSE,
    delete = TRUE,
    min = min,
    max = max
  )

  alpha = score_info$alpha
  G6 = score_info$G6

  return(score_info)
  # return(score_info$alpha |> pluck(1))
  # return(score_info)
}

compute_reliabilities <- function(admin_session, scored_session_df) {
  # get nested key
  nested_key = admin_session_to_nested_key(admin_session)

  # get all item IDs
  all_item_ids = extract_all_item_ids(admin_session)

  scale_reliabilities <-
    all_item_ids |> unlist(recursive = F) |>

    # map select df by scale sets of item_ids
    map(., function(x) scored_session_df |>
          dplyr::select(all_of(x)) |>

          drop_na() |>

          # assume each scale is unidimensional
          omega(nfactors = 1, plot = FALSE) |>

          # only keep alpha, G6, and Omega total
          keep(names(.) %in% c("alpha", "G6", "omega.tot"))) |>

    # set names of outputed lists to scale names
    set_names(names(nested_key |> unlist(recursive = F))) |>

    # convert to data.frame to keep row labels
    map(unlist) |>
    as.data.frame() |>

    # suppress warnings and messages (we are ignoring the other omegas anyway)
    suppressWarnings() |> suppressMessages() |>
    t() |> as.data.frame() |> rownames_to_column(var = "Scale")

  return(scale_reliabilities)
}