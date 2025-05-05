library(dplyr)
# summary_data <- read.csv("summary.csv")
raw_summary_data <-  read.csv("summary.csv") %>%
    select(instance,config,struct,seed,best,runtime_ms) %>%
    mutate(instance = sub("_.*","",instance),
           config2=sub("\\..*","",config),
           num_clusters = gsub("[a-z ]*","",config2),
           type=gsub("[0-9 ]*","",config2),
           chunks = gsub("[^0-9]","",struct),
           algorithm = ifelse(grepl("gaia",struct), "gaia","venus"),
           config=NULL,
           config2=NULL,
           struct=NULL) %>%
    filter(num_clusters == 8)

# TODO pivot
raw_summary_data %>% group_by(instance,type,algorithm) %>%
    summarise(best_solution = min(best),
              avg_solution=mean(best),
              standard_dev = sd(best)) %>%
    print(n=32)


library(data.table)
library(dplyr)
library(stringr)
library(readr)
# DT <- list.files("logs/slashdot", "\\.csv$", full.names = TRUE) %>%
#       rbindlist(idcol = "file")        # fread() is default reader inside rbindlist
# full_slashdot_data <- DT[, file := tools::file_path_sans_ext(basename(file))]
# type, num_clusters, algorithm, seed
files <- list.files(path = "logs/bundestag", pattern = "\\.csv$", full.names = TRUE)
full_slashdot_data <- do.call(
  rbind, lapply(files, function(f) {
    df <- read.csv(f, stringsAsFactors = FALSE)
    df$filename <- basename(f)
    df })) %>%
    mutate(
    instance        = word(filename, 1, sep = "_"),
    algorithm       = word(filename, 4, sep = "_"),
    chunks        = word(filename, 6, sep = "_") %>%
                   str_remove("\\.json$") %>%
                   parse_number(),
    type      = word(filename, 7, sep = "_") %>%
                   str_extract("^[A-Za-z]+"),
    num_clusters       = word(filename, 7, sep = "_") %>%
                   str_extract("[0-9]+") %>%
                   parse_number(),
    seed  = str_extract(filename, "(?<=_)(\\d+)(?=\\.csv$)") %>%
                   as.integer()
  )
library(dplyr)
library(tidyr)
library(zoo)
# summary_data <- x %>% group_by(instance,type,num_clusters,chunks,algorithm) %>% summarise(min = min(best), mean = mean(best), median = median(best), sd = sd(best))
filled_data <- full_slashdot_data %>%
    group_by(algorithm, num_clusters, type, seed, chunks) %>%
    mutate(runtime = ceiling(runtime/100)*100) %>%
    complete(seed, runtime = seq(5000, 50000, by = 100)) %>%
    arrange(runtime) %>%
    fill(best) %>%
    filter(!is.na(best)) %>%
    arrange(runtime) %>%
    mutate(obj = cummin(best)) %>%
    ungroup()

library(ggplot2)

theme <- theme_minimal() +
    theme(legend.position = "bottom",
          legend.direction="horizontal",
          legend.text = element_text(size = 16),  # Adjust legend text size
          legend.key.width = unit(0.75, "cm"),  # Adjust width of legend keys (color boxes)
          legend.key.height = unit(0.5, "cm"),
          axis.text = element_text(size = 14),
          axis.title = element_text(size = 16),
          plot.title = element_text(size = 18, face = "bold"),
          strip.text = element_text(size = 16, vjust =2), # Left-align facet labels
          axis.text.y = element_text(size = 14, hjust = 0), # Right-align y-axis labels
          panel.grid.minor = element_blank(),
          # panel.grid.major = element_blank(),
          strip.placement = "outside",
          # Keep the panel border
          # panel.border = element_rect(color = "gray70", fill = NA, size=0.5),
          axis.ticks = element_line(),
          legend.title=element_blank()) # Keep tick marks

#list.files("logs/slashdot", "\\.csv$", full.names = TRUE) %>% rbind()

summary_df <- filled_data %>%
  filter(chunks == 10, num_clusters < 11.9) %>%
  group_by(type, algorithm, num_clusters, runtime) %>%
  summarize(mean_obj = mean(obj, na.rm = TRUE), .groups = "drop")
ggplot(summary_df, aes(x=runtime,y=mean_obj,color = as.factor(num_clusters), linetype=algorithm)) +
    stat_summary_bin(
    fun = mean,            # summary function
    binwidth=100,             # number of bins (or use binwidth = 0.5)
    geom = "line",size=2        # draw as a line
  ) + facet_wrap(.~type) + coord_cartesian(ylim=c(0,20000)) + xlim(0,10000)

