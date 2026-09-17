library(httr2)
library(purrr)
library(dplyr)
library(tidyr)

# 1. Your API Key (fill with your API key)
MY_API_KEY <- ""

# 2. Helper function to generate the 3 possible 10-digit formats
generate_ndc_variants <- function(ndc11) {
  # Remove any existing hyphens/spaces to ensure we have exactly 11 digits
  n <- gsub("[^0-9]", "", as.character(ndc11))
  
  if (nchar(n) != 11) return(NULL)
  
  list(
    v442 = paste0(substr(n, 2, 5), "-", substr(n, 6, 9), "-", substr(n, 10, 11)),
    v532 = paste0(substr(n, 1, 5), "-", substr(n, 7, 9), "-", substr(n, 10, 11)),
    v541 = paste0(substr(n, 1, 5), "-", substr(n, 6, 9), "-", substr(n, 11, 11))
  )
}

# 3. Modified API lookup function
get_ndc_data <- function(ndc11, api_key) {
  variants <- generate_ndc_variants(ndc11)
  
  if (is.null(variants)) {
    return(data.frame(ndc_query = ndc11, brand_name = "INVALID NDC",
                      route = "NA", status = "ERROR"))
  }
  
  base_url <- "https://api.fda.gov/drug/ndc.json"
  results_list <- list()
  
  # Check each of the 3 variants
  for (fmt_name in names(variants)) {
    ndc_to_search <- variants[[fmt_name]]
    
    req <- request(base_url) %>%
      req_url_query(
        api_key = api_key,
        search = paste0("packaging.package_ndc:\"", ndc_to_search, "\""),
        limit = 1
      )
    
    tryCatch({
      resp <- req_perform(req)
      content <- resp_body_json(resp)
      
      if (!is.null(content$results)) {
        res <- content$results[[1]]
        results_list[[fmt_name]] <- data.frame(
          brand_name = ifelse(is.null(res$brand_name), "GENERIC", res$brand_name),
          route = paste(unlist(res$route), collapse = ", "),
          found_via = fmt_name,
          stringsAsFactors = FALSE
        )
      }
    }, error = function(e) {
      # 404 or connection issues: do nothing, just move to the next variant
    })
  }
  
  # Logic to handle results and flagging
  num_found <- length(results_list)
  
  if (num_found == 0) {
    return(data.frame(ndc_query = ndc11, brand_name = "NOT FOUND", route = "NA",
                      status = "NOT FOUND"))
  } else if (num_found == 1) {
    res_df <- results_list[[1]]
    return(data.frame(ndc_query = ndc11, brand_name = res_df$brand_name, 
                      route = res_df$route, status = paste("OK:", res_df$found_via)))
  } else {
    # FLAG: Multiple variants returned data
    # Combine the routes found to show the conflict
    all_routes <- map_chr(results_list, ~ .x$route) %>% unique() %>% paste(collapse = " | ")
    return(data.frame(ndc_query = ndc11, brand_name = "MULTIPLE HITS",
                      route = all_routes, status = "FLAG: MANUAL CHECK"))
  }
}

# 4. EXECUTION
meps = read.csv("meps_2022_to_api.csv") # replace by CSV of user's choice with 
                                        # NDC as one column
meps = meps %>% 
  drop_na(NDC)
unique_ndcs <- unique(meps$NDC)

# Use map_df to lookup each unique 11-digit NDC
lookup_results <- map_df(unique_ndcs, ~get_ndc_data(.x, MY_API_KEY))

# 5. JOIN: Attach results back to the original df
meps <- meps %>%
  left_join(lookup_results, by = c("NDC" = "ndc_query"))

print(head(meps))