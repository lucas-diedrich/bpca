#!/usr/bin/env Rscript
# Run BPCA on test cases using pcaMethods::bpca.
# This script reads JSON test case files and writes results to JSON.

library(pcaMethods)
library(jsonlite)

#' Run BPCA using the raw bpca() function.
#'
#' @param X Matrix with observations in rows and variables in columns.
#'   May contain NA values.
#' @param n_latent Number of principal components to compute.
#' @param max_iter Maximum number of iterations.
#' @param threshold Convergence threshold for log10(tau) change.
#' @return List containing scores and loadings.
run_bpca <- function(X, n_latent, max_iter = 1000, threshold = 1e-4) {
  # Use raw bpca() function directly - it handles centering internally
  # This matches how the Python implementation works
  result <- bpca(
    X,
    nPcs = n_latent,
    maxSteps = max_iter,
    threshold = threshold,
    verbose = FALSE
  )

  list(
    scores = scores(result),      # (n_obs, n_latent)
    loadings = loadings(result)   # (n_var, n_latent)
  )
}


#' Process a single test case file.
#'
#' @param input_path Path to input JSON file containing test case.
#' @param output_path Path to output JSON file for results.
process_case <- function(input_path, output_path) {
  # Read test case (simplifyMatrix = FALSE to handle null values properly)
  test_case <- fromJSON(input_path, simplifyMatrix = FALSE)

  # Convert X to matrix (null values become NA in R)
  # Each element of test_case$X is a row (list of values)
  X <- matrix(
    unlist(lapply(test_case$X, function(row) {
      sapply(row, function(v) if (is.null(v)) NA else v)
    })),
    nrow = test_case$n_obs,
    ncol = test_case$n_var,
    byrow = TRUE
  )

  # Run BPCA
  result <- run_bpca(
    X,
    n_latent = test_case$n_latent,
    max_iter = test_case$max_iter,
    threshold = test_case$tolerance
  )

  # Prepare output
  output <- list(
    case_id = test_case$case_id,
    scores = as.list(as.data.frame(result$scores)),
    loadings = as.list(as.data.frame(result$loadings))
  )

  # Write result
  write_json(output, output_path, auto_unbox = TRUE, digits = 15)
}


#' Process all test cases in a directory.
#'
#' @param input_dir Directory containing test case JSON files.
#' @param output_dir Directory to write result JSON files.
process_directory <- function(input_dir, output_dir) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  input_files <- list.files(input_dir, pattern = "^case_.*\\.json$", full.names = TRUE)
  cat(sprintf("Found %d test case files\n", length(input_files)))

  for (input_path in input_files) {
    case_name <- sub("\\.json$", "", basename(input_path))
    output_path <- file.path(output_dir, paste0(case_name, "_r_result.json"))

    tryCatch({
      process_case(input_path, output_path)
      cat(sprintf("Processed: %s\n", case_name))
    }, error = function(e) {
      cat(sprintf("ERROR processing %s: %s\n", case_name, e$message))
    })
  }
}


# Main entry point
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  if (length(args) < 2) {
    cat("Usage: Rscript run_bpca_r.R <input_dir> <output_dir>\n")
    cat("\nProcesses all case_*.json files in input_dir and writes results to output_dir.\n")
    quit(status = 1)
  }

  input_dir <- args[1]
  output_dir <- args[2]

  cat(sprintf("Input directory: %s\n", input_dir))
  cat(sprintf("Output directory: %s\n", output_dir))

  process_directory(input_dir, output_dir)
  cat("Done.\n")
}

main()
