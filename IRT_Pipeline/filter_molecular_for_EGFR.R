#!/home/chad/opt/R-4.3.2/bin/Rscript


# Load argparse library
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
parser$add_argument('--completed', required = FALSE  , default = 'final_file_aggregate_slide_id.csv', help = 'csv file allows for filtering of cases that have already been processed')
# Parse the arguments
args <- parser$parse_args()

# Use the arguments in your script
input_file <- args$input_csv
output_dir <- args$directory_out
filter_criteria <- args$test_to_filter
# Assuming args is a list or environment containing the 'completed' element
args$completed <- "/media/hdd1/chad/production/run_EGFR/EGFR_results/gigapath_results.csv"

if (!is.null(args$completed)) {
  completed_file <- args$completed
  
  # Perform actions if 'completed' is present
  print(paste("Completed file:", completed_file))
  completed_cases <- read.table(completed_file, sep = ",",  header = TRUE,  stringsAsFactors = FALSE)
  
  
  
} else {
  print("The 'completed' argument is not present.")
  completed_cases <- data.frame() %>% mutate(slide_id = NA)
  
}

input_file <- "/media/hdd1/chad/production/run_EGFR/molecular_watcher/Mnumber_Snumber_block.csv"
filter_criteria <- "EGFR via Idylla"
all_cases <- read.csv(input_file, stringsAsFactors = FALSE)


idylla_cases <- all_cases %>% filter(str_detect(category_description, filter_criteria), Snumber != "") 

idylla_cases_to_merge <- idylla_cases %>%
select(Mnumber, Snumber, block_id) %>%
rename(IDYLLA_Mnumber = Mnumber)  

impact_cases <- all_cases %>%
  filter(category_description == "IMPACT-Solid TUMOR", Snumber != "") %>%
  select(Mnumber, Snumber, block_id) %>%
  rename(IMPACT_Mnumber = Mnumber) %>%
  full_join(idylla_cases_to_merge ,
    by = c("block_id" = "block_id", "Snumber" = "Snumber"), 
    relationship = "many-to-many") %>%
  filter(!is.na(IDYLLA_Mnumber))

# Define the directory and pattern
file_list <- system("find /home/chad/production/slide_data/* -name 'manifest_*.csv'", intern = TRUE)

# /home/chad/production/slide_data/2024-04-28/manifest_2024-04-28.csv
# Read and bind all files into a single dataframe
all_data <- do.call(rbind, lapply(file_list, function(file) {
  data <- read.csv(file, stringsAsFactors = FALSE)
  data <- data[, 1:13]  # Keep only the first 13 columns
  return(data)
}))
combined_vector <- c(impact_cases$IMPACT_Mnumber, impact_cases$IDYLLA_Mnumber, idylla_cases$Snumber)
cleaned_vector <- combined_vector[!is.na(combined_vector)]
all_data_filter_eligible <- all_data %>%
  filter(str_detect(tolower(stain), "h&e") ) %>%
  filter(case_id_slide %in% cleaned_vector) %>%
  mutate(part_block=paste0(part_id, "-", str_remove_all(block_name, " "))) %>%
  distinct(slide_file_name, .keep_all = TRUE) 


final_file <- impact_cases %>%
mutate(part_block = str_remove_all(block_id, " ")) %>%
mutate(IMPACT_slide_id = NA) %>%
mutate(IDYLLA_slide_id = NA) %>%
mutate(Snumber_slide_id = NA)
i <- 800
slides_to_processes <- NULL
for (i in 1:nrow(final_file)) {
    if (i %% 20 == 0) {
      print(paste0("Processing case ", i, " of ", nrow(final_file)))
    }
    Snumber <- final_file$Snumber[i]
    IDYLLA_Mnumber <- final_file$IDYLLA_Mnumber[i]
    IMPACT_Mnumber <- final_file$IMPACT_Mnumber[i]
    part_block_intern <- final_file$part_block[i]
    # process each idylla case
    idylla_slide <- all_data_filter_eligible %>%
    filter(case_id_slide == IDYLLA_Mnumber) %>%
    mutate(file_path = NA, match_status=NA) %>%
    select(slide_file_name,sub_specialty,case_accessionDate,case_id_slide,part_id,block_name,slide_barcode,slide_id,scanner_id,Molecular_Block,part_block,match_status,file_path)
    final_file$IDYLLA_slide_id[i] <- paste0(idylla_slide$slide_id, collapse = "|")
    # process each impact case
    impact_slide <- all_data_filter_eligible %>%
    filter(case_id_slide == IMPACT_Mnumber) %>%
    mutate(file_path = NA, match_status=NA) %>%
    select(slide_file_name,sub_specialty,case_accessionDate,case_id_slide,part_id,block_name,slide_barcode,slide_id,scanner_id,Molecular_Block,part_block,match_status,file_path)
    final_file$IMPACT_slide_id[i] <- paste0(impact_slide$slide_id, collapse = "|")
    # process each Snumber case
    Snumber_slide <- all_data_filter_eligible %>%
    filter(case_id_slide == Snumber & part_block == part_block_intern) %>%
    mutate(file_path = NA, match_status=NA) %>%
    select(slide_file_name,sub_specialty,case_accessionDate,case_id_slide,part_id,block_name,slide_barcode,slide_id,scanner_id,Molecular_Block,part_block,match_status,file_path)
    final_file$Snumber_slide_id[i] <- paste0(Snumber_slide$slide_id, collapse = "|")
    slides_to_processes <- rbind(slides_to_processes, idylla_slide, impact_slide, Snumber_slide)
}

final_file %>% write.table(file =  paste0(output_dir, "/final_file_aggregate_slide_id.csv"), sep = ",", row.names = FALSE, col.names = TRUE)

slides_to_processes %>%filter(slide_id %in% completed_cases$slide_id == FALSE) %>%
  write.table(file = paste0(output_dir, "/slides_to_processes.csv"), sep = ",", row.names = FALSE, col.names = TRUE)
