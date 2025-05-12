library(tidyverse)

synthetic_data <- rbind(read.csv("./synthetic_summary_w_accuracy.csv"),read.csv("./synthetic_summary_2_w_accuracy.csv")) %>% mutate(
  ## from the instance filename
  n_per_int   = str_extract(instance, "(?<=n-per-int_)\\d+")              %>% as.integer(),
  m_frac      = str_extract(instance, "(?<=m-frac_)\\d+\\.?\\d*")         %>% parse_number(),
  p           = str_extract(instance, "(?<=_p_)\\d+\\.?\\d*")            %>% parse_number(),
  disagreement= str_extract(instance, "(?<=disagreement_)\\d+")           %>% as.integer(),
  ## from the config filename
  config_id   = str_extract(config, "(?<=config_)[^_]+") ,
  chunks      = str_extract(config, "(?<=chunks_)\\d+")                  %>% as.integer()) %>%
  mutate(p=factor(1-p,levels=c("0","0.1","0.2","0.3","0.4","0.5")),
         algorithm = factor(config_id,levels=c("gaia","venus"), labels=c("Gaia","Venus")))


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
        # legend.title=element_blank()
  ) # Keep tick marks



# g <- ggplot(synthetic_data %>% filter(m_frac %in% c(0.01,0.1,1), chunks==10)) +
#   geom_boxplot(aes(x=p,
#                    y=(best-disagreement)/(edge_weight)*100,
#                    fill=algorithm,
#                    # color = as.factor(chunks),
#                    group    = interaction(p, algorithm, chunks))) +
#   # scale_colour_manual(values=c("10"="#222222","100"="#CCCCCC"))+
#   theme +
#   scale_fill_manual(values=c("Gaia"="#00BFC4","Venus"="#F8766D")) +
#   labs(
#     fill   = "Algorithm",   # legend for fill
#     colour = "Number of Batches",     # legend for outline colour
#     y= "Normalized Objective versus Ground Truth",
#     x= "Label Flip Probability"
#   )  + facet_wrap(.~m_frac,labeller = labeller(m_frac=c("0.01"="Density: 0.01","0.1"="Density: 0.1","1"="Density: 1")))
library(lemon)

g <- ggplot(synthetic_data %>% filter(m_frac %in% c(0.01,0.1,1), chunks==10),
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
  facet_rep_wrap(.~m_frac,labeller = labeller(m_frac=c("0.01"="Density: 0.01","0.1"="Density: 0.1","1"="Density: 1")),repeat.tick.labels = TRUE) +
  theme + theme(repeat.tick.labels = TRUE) +
  scale_fill_manual(values=c("Gaia"="#00BFC4","Venus"="#F8766D")) +
  scale_color_manual(values=c("Gaia"="#00BFC4","Venus"="#F8766D")) +
  labs(fill   = "Algorithm",   # legend for fill
       colour = "Algorithm",     # legend for outline colour
       y= "Percentage Point Δ vs Ground Truth (lower is better)",
       x= "Label Flip Probability")

ggsave(filename = "synth_objective_plot2.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 16,                 # in inches (adjust to taste)
       height   = 6,                  # in inches
       units    = "in")

g <- ggplot(synthetic_data %>%
            # group_by(p,algorithm,chunks,m_frac) %>%
            # summarize(accuracy=mean(accuracy)) %>%
            filter(m_frac %in% c(0.01,0.1,1), chunks==10),
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
  facet_rep_wrap(.~m_frac,labeller = labeller(m_frac=c("0.01"="Density: 0.01","0.1"="Density: 0.1","1"="Density: 1")),repeat.tick.labels = TRUE) +
  coord_cartesian(ylim=c(0.125,1)) +
  geom_hline(aes(yintercept=0.125, color="Random Assignment", fill="Random Assignment"),show.legend=T)+
  scale_color_manual(values=c("Gaia"="#00BFC4","Venus"="#F8766D", "Random Assignment"="#000000")) +
  scale_fill_manual(values=c("Gaia"="#00BFC4","Venus"="#F8766D", "Random Assignment"="#000000")) +
  labs(
    linetype   = "Number of Batches",   # legend for fill
    colour = "Algorithm",     # legend for outline colour05
    fill = "Algorithm",     # legend for outline colour05
    y= "Vertex Assignment Accuracy",
    x= "Label Flip Probability"
  ) + theme
ggsave(filename = "synth_accuracy_plot2.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 16,                 # in inches (adjust to taste)
       height   = 6,                  # in inches
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
    x= "Label Flip Probability"
  ) + facet_wrap(.~m_frac,labeller = labeller(m_frac=c("0.01"="Density: 0.01","0.1"="Density: 0.1","1"="Density: 1"))) +



ggsave(filename = "synth_accuracy_plot.pdf",
       plot     = g,        # or explicitly your ggplot object3000F", "#FFED00", "#151518", "#009EE0")) +
       device   = cairo_pdf,          # better text handling & UTF-8 supporttheme + scale_linewidth_identity() +guides()
       width    = 16,                 # in inches (adjust to taste)
       height   = 6,                  # in inches
       units    = "in")


