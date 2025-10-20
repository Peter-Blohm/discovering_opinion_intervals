library(tidyverse)

synthetic_data <- rbind(read.csv("./synthetic_summary_3_w_accuracy.csv")) %>% mutate(
  ## from the instance filename
  n_per_int   = str_extract(instance, "(?<=n-per-int_)\\d+")              %>% as.integer(),
  m_frac      = str_extract(instance, "(?<=m-frac_)\\d+\\.?\\d*")         %>% parse_number(),
  p           = str_extract(instance, "(?<=_p_)\\d+\\.?\\d*")            %>% parse_number(),
  disagreement= str_extract(instance, "(?<=disagreement_)\\d+")           %>% as.integer(),
  ## from the config filename
  config_id   = str_extract(config, "(?<=config_)[^_]+") ,
  chunks      = str_extract(config, "(?<=chunks_)\\d+")                  %>% as.integer()) %>%
  mutate(p=factor(1-p,levels=c("0","0.1","0.2","0.3","0.4","0.5")),
         algorithm = factor(config_id,levels=c("gaia","venus"), labels=c("GAIA","VENUS")))


theme <- theme_minimal() +
  theme(legend.position = "bottom",
        # legend.direction="horizontal",
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
        legend.title=element_text(size=16),
  ) # Keep tick marks



# g <- ggplot(synthetic_data %>% filter(m_frac %in% c(0.01,0.1,1), chunks==10)) +
#   geom_boxplot(aes(x=p,
#                    y=(best-disagreement)/(edge_weight)*100,
#                    fill=algorithm,
#                    # color = as.factor(chunks),
#                    group    = interaction(p, algorithm, chunks))) +
#   # scale_colour_manual(values=c("10"="#222222","100"="#CCCCCC"))+
#   theme +
#   scale_fill_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
#   labs(
#     fill   = "Algorithm",   # legend for fill
#     colour = "Number of Batches",     # legend for outline colour
#     y= "Normalized Objective versus Ground Truth",
#     x= "Sign Flip Probability"
#   )  + facet_wrap(.~m_frac,labeller = labeller(m_frac=c("0.01"="Density: 0.01","0.1"="Density: 0.1","1"="Density: 1")))
library(lemon)
library(ggh4x)

my_labeller <- as_labeller(function(x) {
  paste0(
    "Sign Flip Probability\n",
    "Density: ", x
  )
})
########################################### THIS ##################################################
g <- ggplot(synthetic_data %>% filter(m_frac %in% c(0.01,0.03,0.1,1), chunks==10),
       aes(x=p,
           y=(best-disagreement)/(edge_weight)*100,
           color=algorithm,
           fill=algorithm,
           group=algorithm)) +
  stat_summary(fun = mean,            # one number per unique x
             geom = "line",
             linewidth = 1) +
  ## 2 · SD ribbon ───────────────────────────────────────────
  stat_summary(fun.data  = mean_sdl,  # returns ymin = μ-σ, y = μ, ymax = μ+σ
               fun.args  = list(mult = 1),   # 1 = ±1 SD; use 2 for ±2 SD, …
               geom      = "ribbon",
               alpha     = .25,
               colour    = NA)  +
  theme + theme(legend = element_blank()) +
  # theme(legend.position= c(0.25,0.175),
  #       legend.frame = element_rect(colour = "white")) + #.inside=c(0.5,0.51))
  scale_fill_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  scale_color_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  labs(fill   = NULL,   # legend for fill
       colour = NULL,     # legend for outline colour
       y= "Difference (%)",
       x= NULL) + ylim(-40,10) +
  facet_grid2(.~m_frac, labeller = labeller(m_frac = my_labeller),
              scales = "free_x",      # if you want each panel's x scale independent
    axes   = "all",         # draw axes (ticks + labels) on all panels
    switch = "x") + theme(legend.position = "none")
# labeller = labeller(m_frac=c("0.01"="Density: 0.01","0.03"="Density: 0.03","0.1"="Density: 0.1","1"="Density: 1"))
ggsave(filename = "synth_plot_all_objective.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 5.5 * 1.5,                 # in inches (adjust to taste)
       height   = 1.25 * 1.5,                  # in inches
       units    = "in")

# synthetic_data %>% group_by(algorithm,m_frac,p,structure) %>% arrange(desc(best),desc(accuracy)) %>% filter(row_number()==1) %>%

facet_rep_grid(.~m_frac,labeller =  ,repeat.tick.labels = TRUE) +
  theme + theme(repeat.tick.labels = TRUE) +
  scale_fill_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  scale_color_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  labs(fill   = "Algorithm",   # legend for fill
       colour = "Algorithm",     # legend for outline colour
       y= "Percentage Point Δ vs Ground Truth (lower is better)",
       x= "Sign Flip Probability")

ggsave(filename = "synth_objective_plot2.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 16,                 # in inches (adjust to taste)
       height   = 6,                  # in inches
       units    = "in")
 # %>% group_by(algorithm,m_frac,p,chunks) %>% arrange(desc(best),accuracy) %>% filter(row_number()==n())
g <- ggplot(synthetic_data%>%
              group_by(algorithm,m_frac,p,chunks) %>%
              filter(best == min(best)) %>%
              filter(m_frac %in% c(0.01,0.03,0.1,1)) %>%
            # group_by(p,algorithm,chunks,m_frac) %>%
            # summarize(accuracy=mean(accuracy)) %>%
            filter(chunks==10),
            aes(x=p,
                y=accuracy,
                color=algorithm,
                fill=algorithm,
                group=algorithm)) +
  stat_summary(fun = mean,            # one number per unique x
               geom = "line",
               linewidth = 1) +
  ## 2 · SD ribbon ───────────────────────────────────────────
  stat_summary(fun.data  = mean_sdl,  # returns ymin = μ-σ, y = μ, ymax = μ+σ
               fun.args  = list(mult = 1),   # 1 = ±1 SD; use 2 for ±2 SD, …
               geom      = "ribbon",
               alpha     = .25,
               colour    = NA) +
  facet_grid2(.~m_frac,labeller = labeller(m_frac=my_labeller),
              scales = "free_x",      # if you want each panel's x scale independent
    axes   = "all",         # draw axes (ticks + labels) on all panels
    switch = "x") +
  coord_cartesian(ylim=c(0.0,1)) +
  geom_hline(aes(yintercept=0.125, color="Random Assignment", fill="Random Assignment"),show.legend=T)+
  scale_color_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D", "Random Assignment"="#000000")) +
  scale_fill_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D", "Random Assignment"="#000000")) +
  labs(
    linetype   = "Number of Batches",   # legend for fill
    colour = "Algorithm",     # legend for outline colour05
    fill = "Algorithm",     # legend for outline colour05
    y= "Accuracy",
    x= NULL
  ) + theme + theme(legend.position = "bottom",
legend.margin        = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"),

    # **Legend box margin**: space between stacked legends (if legend.box="vertical")
    legend.box.margin    = margin(t = 0, r = 0, b = 0, l = 0, unit = "cm"),

    # **Spacing between legend keys** (horizontal and vertical)
    legend.spacing.x     = unit(0, "cm"),
    legend.spacing.y     = unit(0.1, "cm"),
    # **Padding inside each key** (controls distance between symbol and key border)
    # legend.key_PADDING   = unit(0.0, "cm"),

    # **Key size** (height/width of each legend key box)
    legend.key.height    = unit(0.3, "cm"),
    legend.key.width     = unit(0.8, "cm"),
                    plot.margin = margin(t = 5, r = 5, b = 5, l = 5, unit = "pt")
  )
ggsave(filename = "synth_all_accuracy_plot_best.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 5.5 * 1.5,                 # in inches (adjust to taste)
       height   = 1.5 * 1.5,                  # in inches
       units    = "in")
g

g <- ggplot(#synthetic_data %>%
            # group_by(p,algorithm,chunks,m_frac) %>%
            # summarize(accuracy=mean(accuracy)) %>%
            # filter(m_frac==1,chunks==10),
            synthetic_data %>% group_by(algorithm,m_frac,p) %>% arrange(desc(best),desc(accuracy)) %>% filter(row_number()==1),

            aes(x=p,
                y=accuracy,
                color=algorithm,
                fill=algorithm,
                group=algorithm)) +
  stat_summary(fun = mean,            # one number per unique x
               geom = "line",
               linewidth = 1) +
  ## 2 · SD ribbon ───────────────────────────────────────────
  stat_summary(fun.data  = mean_sdl,  # returns ymin = μ-σ, y = μ, ymax = μ+σ
               fun.args  = list(mult = 1),   # 1 = ±1 SD; use 2 for ±2 SD, …
               geom      = "ribbon",
               alpha     = .25,
               colour    = NA) +
  facet_rep_wrap(.~m_frac,labeller = labeller(m_frac=my_labeller)) +
  coord_cartesian(ylim=c(0.0,1)) +
  geom_hline(aes(yintercept=0.125, color="Random", fill="Random"),show.legend=T)+
  scale_color_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D", "Random"="#000000")) +
  scale_fill_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D", "Random"="#000000")) +
  labs(
    linetype   = "Number of Batches",   # legend for fill
    colour = "Algorithm",     # legend for outline colour05
    fill = "Algorithm",     # legend for outline colour05
    y= "Vertex Assignment Accuracy",
    x= "Sign Flip Probability"
  ) + theme +
  theme(legend.position= c(1-0.25,1-0.175),
        legend.frame = element_rect(colour = "white")) + #.inside=c(0.5,0.51))
  # scale_fill_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  # scale_color_manual(values=c("GAIA"="#00BFC4","VENUS"="#F8766D")) +
  labs(fill   = "Algorithm",   # legend for fill
       colour = "Algorithm",     # legend for outline colour
       y= "Accuracy",
       x= "Sign Flip Probability")
g

ggsave(filename = "synth_accuracy_plot21.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 4,                 # in inches (adjust to taste)
       height   = 3.5,                  # in inches
       units    = "in")



geom_line(aes(x = p,
                y = accuracy,
                color = algorithm,
                linetype = as.factor(chunks),
                group = interaction(algorithm,chunks)),
            linewidth = 1.5) + theme +
  labs(
    linetype   = "Number of Batches",   # legend for fill
    colour = "Algorithm",     # legend for outline colour05
    y= "Objective versus Ground Truth",
    x= "Sign Flip Probability"
  ) + facet_wrap(.~m_frac,labeller = labeller(m_frac=c("0.01"="Density: 0.01","0.1"="Density: 0.1","1"="Density: 1"))) +



ggsave(filename = "synth_accuracy_plot.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 16,                 # in inches (adjust to taste)
       height   = 6,                  # in inches
       units    = "in")


