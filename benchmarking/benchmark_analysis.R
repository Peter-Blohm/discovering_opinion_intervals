library(dplyr)
# summary_data <- read.csv("summary.csv")
raw_summary_data <-  read.csv("summary_2.csv") %>%
    select(instance,config,struct,seed,best,runtime_ms) %>%
    mutate(instance = sub("_.*","",instance),
           config2=sub("\\..*","",config),
           num_clusters = gsub("[a-z ]*","",config2),
           type=gsub("[0-9 ]*","",config2),
           chunks = gsub("[^0-9]","",struct),
           algorithm = ifelse(grepl("gaia",struct), "gaia","venus"),
           config=NULL,
           config2=NULL,
           struct=NULL,
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
           instance = case_when(
             instance == "bitcoinotc"      ~ "\\Bitcoin",
             instance == "bundestag"       ~ "\\Bundestag",
             instance == "chess"       ~ "\\Chess",
             instance == "elec"       ~ "\\WikiElec",
             instance == "epinions"       ~ "\\Epinions",
             instance == "slashdot"       ~ "\\Slashdot",
             instance == "wikiconflict"       ~ "\\WikiConflict",
             instance == "wikisigned-k2"       ~ "\\WikiSigned",
           )
    ) %>%
    filter(num_clusters == 8, type=="intervals", chunks==10)



# TODO pivot
library(kableExtra)

myformat <- function(x) format(round(x),big.mark="\\\\,",trim= TRUE)

raw_summary_data %>% group_by(instance,algorithm) %>%
    summarise(best_solution = min(best),
              avg_solution=mean(best),
              standard_dev = sd(best)) %>%
    pivot_wider(names_from=algorithm,values_from=c(best_solution,avg_solution,standard_dev)) %>%
    ungroup() %>%
    transmute(dataset=instance,gaia_best=best_solution_gaia,gaia_avg=paste(myformat(avg_solution_gaia),"p",myformat(standard_dev_gaia)),
              venus_best=best_solution_venus,venus_avg=paste(myformat(avg_solution_venus),"p",myformat(standard_dev_venus))) %>%
    kable(format    = "latex",
          booktabs  = TRUE,
          escape    = FALSE,
          format.args = list(big.mark = "\\\\,"),
          col.names = c("Dataset", "Best",   "\\text{Avg}p\\text{Std}", "Best",   "\\text{Avg}p\\text{Std}"),
          align     = c("l", "r", "D{p}{\\pm}{6.4}", "r", "D{p}{\\pm}{6.4}"))%>%
    collapse_rows(columns = 1, row_group_label_position= "identity", latex_hline              = "none") %>%
    add_header_above(c(" " = 1, "\\texttt{GAIA}" = 2, "\\texttt{VENUS}" = 2),escape=F) %>%
    kable_styling(latex_options = c("hold_position")) %>%
    as.character() %>%
    cat(sep = "\n")

    print(n=32)


library(data.table)
library(dplyr)
library(stringr)
library(readr)
# DT <- list.files("logs/slashdot", "\\.csv$", full.names = TRUE) %>%
#       rbindlist(idcol = "file")        # fread() is default reader inside rbindlist
# full_slashdot_data <- DT[, file := tools::file_path_sans_ext(basename(file))]
# type, num_clusters, algorithm, seed
files <- list.files(path = "logs2/slashdot", pattern = "\\.csv$", full.names = TRUE)
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
    group_by(algorithm, num_clusters, type, seed, chunks,runtime) %>%
    filter(row_number()==1) %>%
    group_by(algorithm, num_clusters, type, seed, chunks) %>%
    complete(runtime = seq(0, 130000, by=100)) %>%
    arrange(runtime) %>%
    fill(best,.direction="downup") %>%
    filter(!is.na(best)) %>%
    arrange(runtime) %>%
    mutate(obj = cummin(best)) %>%
    ungroup()

library(ggplot2)

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

summary_df <- filled_data %>%
  filter(chunks == 10, num_clusters %% 4 == 0, type=="intervals") %>%
  group_by(algorithm, num_clusters, runtime) %>%
  summarize(mean_obj = mean(obj, na.rm = TRUE),std_obj = sd(obj, na.rm = TRUE), .groups = "drop")
g <- ggplot(summary_df %>%
              mutate(algorithm = factor(algorithm,levels=c("gaia","venus"),labels=c("GAIA","VENUS"))),
            aes(x=runtime,y=mean_obj,color = algorithm, group=interaction(num_clusters,algorithm))) +
  geom_line(aes(linetype=as.factor(num_clusters)),linewidth=0.8)+
  geom_ribbon(aes(ymin = mean_obj - std_obj,max = mean_obj + std_obj,fill=algorithm,color=NULL,group=interaction(num_clusters,algorithm)),alpha=0.2)+
  scale_y_continuous(
    labels = label_number(
      scale    = 1e-3,   # divide values by 1,000
      suffix   = "k",    # add "k"
      accuracy = 1       # rounds to whole numbers (e.g. 120k not 120.4k)
    )
  ) +
  scale_x_continuous(
    labels = label_number(
      scale    = 1e-3,   # divide values by 1,000
      suffix   = "s",    # add "k"
      accuracy = 1,       # rounds to whole numbers (e.g. 120k not 120.4k),

    )
  ) +
  coord_cartesian(ylim=c(42500,57500),xlim=c(0,120000)) +
  scale_color_manual(name= "Algorithm", values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  scale_fill_manual(name="Algorithm",values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  scale_linetype_manual("Number of Intervals", values=c(`4`="solid",`8`="dashed",`12`="dotdash",`16`="dotted"))+
  theme + labs(x="Runtime",y="Disagreement") +guides(
    color     = guide_legend(order = 1, nrow = 1),
    fill      = guide_legend(order = 1, nrow = 1),
    linetype  = guide_legend(order = 2, nrow = 1)
  ) + theme(legend.box = "vertical",legend.title = element_text(size=10),
legend.margin        = margin(t = 0, r = 0, b = 0, l = 0, unit = "cm"),

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



ggsave(filename = "slashdot_time_plot.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 2.7*1.5,                 # in inches (adjust to taste)
       height   = 1.5*1.5,                  # in inches
       units    = "in")




g <- ggplot(full_slashdot_data %>% filter(chunks == 10, num_clusters %% 4 == 0, type=="intervals") %>%
  group_by(algorithm, num_clusters,seed) %>% arrange(runtime) %>% filter(row_number()==n()) %>% mutate(algorithm = factor(algorithm,levels=c("gaia","venus"),labels=c("GAIA","VENUS"))),
       aes(y=best,fill=algorithm,x=as.factor(num_clusters),group=interaction(algorithm,num_clusters))) + geom_boxplot(notch=TRUE) + theme + scale_fill_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  coord_cartesian(ylim=c(30000,40000)) + labs(x="Number of Intervals",y="Disagreement at Convergence") +
  scale_y_continuous(
    labels = label_number(
      scale    = 1e-3,   # divide values by 1,000
      suffix   = "k",    # add "k"
      accuracy = 1       # rounds to whole numbers (e.g. 120k not 120.4k)
    )
  ) +
  coord_cartesian(ylim=c(42500,57500)) +
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

ggsave(filename = "slashdot_conv_plot.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 2.7*1.5,                 # in inches (adjust to taste)
       height   = 1.5*1.5,                  # in inches
       units    = "in")


ggsave(filename = "slashdot_time_plot.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 2.7*1.5,                 # in inches (adjust to taste)
       height   = 2*1.5,                  # in inches
       units    = "in")





x <- raw_summary_data %>%
  group_by(instance,num_clusters,type,algorithm,seed,edges,vertices) %>%
  arrange(runtime_ms) %>%
  summarize(runtime = max(runtime_ms), objective = min(best)) %>%
  group_by(instance,num_clusters,type,algorithm,edges,vertices) %>%
  summarize(runtime = mean(runtime), objective = mean(objective))

ggplot(x,aes(x=as.numeric(edges),y=runtime/1000,color=algorithm,fill=algorithm)) + geom_point() + geom_smooth(method="lm",alpha   = 0.2)