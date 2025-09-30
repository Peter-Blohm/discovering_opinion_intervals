library(tidyverse)
library(dplyr)
library(tidyr)
library(zoo)
library(scales)
# summary_data <- read.csv("summary.csv")
raw_summary_data <-
    read.csv("benchmarking/rebuttal_logs/rebuttal_summary_2.csv",col.names = c("filename","edge_weight","current", "best_batch", "best","best_negative_edge_weight", "epochs_since_restart", "current_temp", "runtime_ms")) %>%
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
  ) %>%
    select(instance,algorithm,chunks,type,num_clusters,best,runtime_ms,seed) %>%
    mutate(instance = sub("_.*","",instance),
           algorithm = ifelse(grepl("gaia",algorithm), "GAIA","VENUS"),
           edges = case_when(
             instance == "bitcoinotc"      ~ "21434",
             instance == "bundestag"       ~ "397497",
             instance == "chess"       ~ "32650",
             instance == "elec"       ~ "100355",
             instance == "epinions"       ~ "708507",
             instance == "slashdot"       ~ "498532",
             instance == "wikiconflict"       ~ "2014053",
             instance == "wikisigned-k2"       ~ "712337",
           ),
           vertices = case_when(
             instance == "bitcoinotc"      ~ "5881",
             instance == "bundestag"       ~ "1480",
             instance == "chess"       ~ "7301",
             instance == "elec"       ~ "7115",
             instance == "epinions"       ~ "131580",
             instance == "slashdot"       ~ "138587",
             instance == "wikiconflict"       ~ "116836",
             instance == "wikisigned-k2"       ~ "138587",
           ),
           instance_name = instance,
           instance = case_when(
             instance == "bitcoinotc"      ~ "Bitcoin",
             instance == "bundestag"       ~ "Bundestag",
             instance == "chess"       ~ "Chess",
             instance == "elec"       ~ "WikiElec",
             instance == "epinions"       ~ "Epinions",
             instance == "slashdot"       ~ "Slashdot",
             instance == "wikiconflict"       ~ "WikiConflict",
             instance == "wikisigned-k2"       ~ "WikiSigned",
           )
    )

raw_summary_data %>%
    filter(type=="intervals",chunks==10) %>%
    group_by(instance,algorithm,num_clusters) %>%
    select(seed,best) %>%
    summarise(lowest = min(best), avg = round(mean(best))) %>%
    pivot_wider(names_from=num_clusters, values_from=c(lowest,avg)) %>%
    select(lowest_4,avg_4,lowest_8,avg_8,lowest_12,avg_12,lowest_16,avg_16) %>%
    mutate(instance = substring(instance,2)) %>%
    knitr::kable(format = "latex")

#
# files <- list.files(path = "logs2/*", pattern = "\\.csv$", full.names = TRUE)
# full_slashdot_data <- do.call(
#   rbind, lapply(files, function(f) {
#     df <- read.csv(f, stringsAsFactors = FALSE)
#     df$filename <- basename(f)
#     df })) %>%
#     mutate(
#     instance        = word(filename, 1, sep = "_"),
#     algorithm       = word(filename, 4, sep = "_"),
#     chunks        = word(filename, 6, sep = "_") %>%
#                    str_remove("\\.json$") %>%
#                    parse_number(),
#     type      = word(filename, 7, sep = "_") %>%
#                    str_extract("^[A-Za-z]+"),
#     num_clusters       = word(filename, 7, sep = "_") %>%
#                    str_extract("[0-9]+") %>%
#                    parse_number(),
#     seed  = str_extract(filename, "(?<=_)(\\d+)(?=\\.csv$)") %>%
#                    as.integer()
#   )
#




# filled_data <- full_slashdot_data %>%
#     group_by(algorithm, num_clusters, type, seed, chunks) %>%
#     mutate(runtime = ceiling(runtime/100)*100) %>%
#     group_by(algorithm, num_clusters, type, seed, chunks,runtime) %>%
#     filter(row_number()==1) %>%
#     group_by(algorithm, num_clusters, type, seed, chunks) %>%
#     complete(runtime = seq(0, 130000, by=100)) %>%
#     arrange(runtime) %>%
#     fill(best,.direction="downup") %>%
#     filter(!is.na(best)) %>%
#     arrange(runtime) %>%
#     mutate(obj = cummin(best)) %>%
#     ungroup()



theme <- theme_minimal() +
    theme(legend.position = "bottom",
          legend.direction="horizontal",
          legend.text = element_text(size = 10),  # Adjust legend text size
          legend.key.width = unit(0.75, "cm"),  # Adjust width of legend keys (color boxes)
          legend.key.height = unit(0.5, "cm"),
          axis.text = element_text(size = 10),
          axis.title = element_text(size = 10),
          plot.title = element_text(size = 10, face = "bold"),
          strip.text = element_text(size = 10, vjust =2), # Left-align facet labels
          axis.text.y = element_text(size = 10, hjust = 0), # Right-align y-axis labels
          panel.grid.minor = element_blank(),
          # panel.grid.major = element_blank(),
          strip.placement = "outside",
          # Keep the panel border
          # panel.border = element_rect(color = "gray70", fill = NA, size=0.5),
          axis.ticks = element_line(),
          legend.title=element_blank()) # Keep tick marks

#list.files("logs/slashdot", "\\.csv$", full.names = TRUE) %>% rbind()

g <- ggplot(raw_summary_data %>% filter(chunks==10, type=="intervals"),
       aes(y=best,fill=algorithm,x=as.factor(num_clusters),group=interaction(algorithm,num_clusters))) +
    geom_boxplot(notch=TRUE) +
    theme +
    facet_wrap(.~instance,scales="free",ncol=4)+
    scale_fill_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  # coord_cartesian(ylim=c(30000,40000)) +
    labs(x="Number of Intervals",y="Disagreement at Convergence") +
  scale_y_continuous(
    labels = label_number(
      scale    = 1e-3,   # divide values by 1,000
      suffix   = "k",    # add "k"
      accuracy = .1       # rounds to whole numbers (e.g. 120k not 120.4k)
    )
  )
g_plus <- g +
  # coord_cartesian(ylim=c(42500,57500)) +
  scale_color_manual(name= "Algorithm", values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  scale_fill_manual(name="Algorithm",values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  scale_linetype_manual("Number of Intervals", values=c(`4`="solid",`8`="dashed",`12`="dotdash",`16`="dotted"))+
  theme + labs(x="Number of Intervals",y="Disagreement") +guides(
    color     = guide_legend(order = 1, nrow = 1),
    fill      = guide_legend(order = 1, nrow = 1),
    linetype  = guide_legend(order = 2, nrow = 1)
  ) + theme(legend.box = "vertical",legend.title = element_text(size=10),
legend.margin        = margin(t = 0, r = 0, b = 14, l = 0, unit = "pt"),

    # **Legend box margin**: space between stacked legends (if legend.box="vertical")
    legend.box.margin    = margin(t = -.4, r = 0, b = 0, l = 0, unit = "cm"),

    # **Spacing between legend keys** (horizontal and vertical)
    legend.spacing.x     = unit(0, "cm"),
    legend.spacing.y     = unit(0.1, "cm"),
    # **Padding inside each key** (controls distance between symbol and key border)
    legend.key_PADDING   = unit(0.0, "cm"),

    # **Key size** (height/width of each legend key box)
    legend.key.height    = unit(0.3, "cm"),
    legend.key.width     = unit(0.8, "cm"),
            plot.margin = margin(t = 5, r = 20, b = 5, l = 5, unit = "pt")
  ) +
  guides(linetype  = guide_legend(
                  order    = 2,
                  nrow     = 1,
                  override.aes = list(linewidth = .8)  # increase line width in legend
                  ,keywidth     = unit(.8, "cm")
                ))

g_plus

ggsave("all_algos.pdf", g_plus, width= 9)


summary_df <- filled_data %>%
  filter(chunks == 10, num_clusters %% 4 == 0, type=="intervals") %>%
  group_by(algorithm, num_clusters, runtime) %>%
  summarize(mean_obj = mean(obj, na.rm = TRUE),std_obj = sd(obj, na.rm = TRUE), .groups = "drop")