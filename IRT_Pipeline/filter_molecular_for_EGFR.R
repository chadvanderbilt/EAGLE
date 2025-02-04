#!/home/chad/opt/R-4.3.2/bin/Rscript

#%%
# This script filters and processes molecular diagnostic data, specifically focusing on EGFR tests.
# It reads molecular data, filters relevant cases based on specified criteria, merges data from multiple sources,
# and generates output CSV files for downstream processing.
#
# **Assumptions:**
# - This script assumes that the manifests for the daily slides being scanned in real-time are available in a standardized format.
# - Adjustments for institutional environment will need to be made as needed.
#
# **Script Overview:**
# - Loads molecular data from an input CSV.
# - Filters cases based on EGFR test criteria (default) or user-specified test.
# - Identifies completed cases to avoid redundant processing.
# - Merges data from Idylla and IMPACT tests and retrieves corresponding slide information.
# - Outputs a final CSV with aggregated slide IDs and another CSV for slides that need processing.
#
# **Libraries Used:**
# - argparse: Parses command-line arguments.
# - dplyr: Provides functions for data manipulation.
# - stringr: Facilitates string operations.
# - tidyr: Helps tidy data for analysis.
#
# **Arguments:**
# - --input_csv: Path to the input CSV file containing molecular data.
# - --directory_out: Output directory for the processed results.
# - --test_to_filter: Filter criteria for specific tests (default is 'EGFR').
# - --completed: Optional CSV to filter out already processed cases.
# - --slide_data_dir: Directory containing the slide data manifests.
#
# **Usage Example:**
# Rscript script.R --input_csv path/to/input.csv --directory_out path/to/output --test_to_filter EGFR --slide_data_dir /home/chad/production/slide_data
#%%

# Load necessary libraries
suppressMessages(library(argparse))
suppressMessages(library(dplyr))
suppressMessages(library(stringr))
suppressMessages(library(tidyr))

# Create a parser object
parser <- ArgumentParser(description = 'Filter molecular data for EGFR')

# Add arguments
parser$add_argument('--input_csv', required = TRUE, help = 'Path to the input file')
parser$add_argument('--directory_out', required = TRUE, help = 'Path to the output directory for Snumber, Idylla, IMPACT')
parser$add_argument('--test_to_filter', required = TRUE, default = 'EGFR', help = 'Filter criteria (default: EGFR)')
parser$add_argument('--completed', required = FALSE, default = 'final_file_aggregate_slide_id.csv', help = 'CSV file to filter out already processed cases')
parser$add_argument('--slide_data_dir', required = TRUE, help = 'Directory containing the slide data manifests')

# Parse the arguments
args <- parser$parse_args()

# Assign arguments to variables
input_file <- args$input_csv
output_dir <- args$directory_out
filter_criteria <- args$test_to_filter
completed_file <- args$completed
slide_data_dir <- args$slide_data_dir

# Load completed cases if provided
if (file.exists(completed_file)) {
  print(paste("Completed file:", completed_file))
  completed_cases <- read.table(completed_file, sep = ",", header = TRUE, stringsAsFactors = FALSE)
} else {
  print("No completed file provided or file does not exist.")
  completed_cases <- data.frame() %>% mutate(slide_id = NA)
}

# Load all molecular cases
all_cases <- read.csv(input_file, stringsAsFactors = FALSE)

# Filter Idylla cases
idylla_cases <- all_cases %>% filter(str_detect(category_description, filter_criteria), Snumber != "")
idylla_cases_to_merge <- idylla_cases %>% select(Mnumber, Snumber, block_id) %>% rename(IDYLLA_Mnumber = Mnumber)

# Filter IMPACT cases and merge with Idylla
impact_cases <- all_cases %>%
  filter(category_description == "IMPACT-Solid TUMOR", Snumber != "") %>%
  select(Mnumber, Snumber, block_id) %>%
  rename(IMPACT_Mnumber = Mnumber) %>%
  full_join(idylla_cases_to_merge, by = c("block_id", "Snumber")) %>%
  filter(!is.na(IDYLLA_Mnumber))

# Load slide data from manifests
file_list <- system(paste0("find ", slide_data_dir, "/* -name 'manifest_*.csv'"), intern = TRUE)
all_data <- do.call(rbind, lapply(file_list, function(file) {
  data <- read.csv(file, stringsAsFactors = FALSE)
  data <- data[, 1:13]  # Keep only the first 13 columns
  return(data)
}))

# Filter eligible slides
combined_vector <- c(impact_cases$IMPACT_Mnumber, impact_cases$IDYLLA_Mnumber, idylla_cases$Snumber)
cleaned_vector <- combined_vector[!is.na(combined_vector)]
all_data_filter_eligible <- all_data %>%
  filter(str_detect(tolower(stain), "h&e")) %>%
  filter(case_id_slide %in% cleaned_vector) %>%
  mutate(part_block = paste0(part_id, "-", str_remove_all(block_name, " "))) %>%
  distinct(slide_file_name, .keep_all = TRUE)

# Prepare final file with slide IDs
final_file <- impact_cases %>%
  mutate(part_block = str_remove_all(block_id, " "), IMPACT_slide_id = NA, IDYLLA_slide_id = NA, Snumber_slide_id = NA)

# Match and append slide IDs to the final file
for (i in 1:nrow(final_file)) {
  if (i %% 20 == 0) print(paste0("Processing case ", i, " of ", nrow(final_file)))

  final_file$IDYLLA_slide_id[i] <- paste0(
    all_data_filter_eligible %>% filter(case_id_slide == final_file$IDYLLA_Mnumber[i]) %>% pull(slide_id),
    collapse = "|")

  final_file$IMPACT_slide_id[i] <- paste0(
    all_data_filter_eligible %>% filter(case_id_slide == final_file$IMPACT_Mnumber[i]) %>% pull(slide_id),
    collapse = "|")

  final_file$Snumber_slide_id[i] <- paste0(
    all_data_filter_eligible %>% filter(case_id_slide == final_file$Snumber[i], part_block == final_file$part_block[i]) %>% pull(slide_id),
    collapse = "|")
}

# Write final files
final_file %>% write.table(file = paste0(output_dir, "/final_file_aggregate_slide_id.csv"), sep = ",", row.names = FALSE)

all_data_filter_eligible %>%
  filter(!slide_id %in% completed_cases$slide_id) %>%
  write.table(file = paste0(output_dir, "/slides_to_processes.csv"), sep = ",", row.names = FALSE)
#%%
