library(dplyr)
library(tidyr)
library(seriation)
library(ggforce)
library(ggpattern)
library(ggplot2)

parse_parties <- function(x) {
  ## ensure we are working with plain character vectors
  x <- as.character(x)

  ## list of canonical → regex-alternatives
  rules <- list(
    SPD      = c("SPD", "SOZIALDEMOKRATISCHE\\s+PARTEI\\s+DEUTSCHLANDS?"),
    CDU      = c("CDU", "CHRISTLICH[ -]?DEMOKRATISCHE[ -]?UNION"),
    CSU      = c("CSU", "CHRISTLICH[ -]?SOZIALE[ -]?UNION"),
    `CDU/CSU`= c("CDU/CSU", "UNION"),                 # Fraktionsebene
    GRÜNE    = c("BÜNDNIS\\s*90\\s*/?\\s*DIE\\s*GRÜNEN?",
                 "BÜ90/GR", "GRÜN[E]?", "GRUENE", "BÜNDNIS`90/DIE GRÜNEN"),
    FDP      = c("FDP", "FREIE[ -]?DEMOKRATISCHE[ -]?PARTEI"),
    LINKE    = c("DIE\\s+LINKE\\.?","LINKS?PARTEI(?:\\.PDS)?","LINKE"),
    AFD      = c("AFD", "ALTERNATIVE[ -]?FÜR[ -]?DEUTSCHLAND"),
    BSW      = c("BSW"),
    fraktionslos = c("[fF]raktionslose?")
  )

  ## start with clean slate
  out <- x # rep(NA_character_, length(x))

  ## try each rule; the first match wins
  for (canon in names(rules)) {
    pat <- paste0("^\\s*(?:", paste(rules[[canon]], collapse = "|"), ")\\.?\\s*$")
    hits <- grepl(pat, x, ignore.case = TRUE, perl = TRUE)
    out[hits] <- canon
  }

  out
}


# cluster weg, runtime dazu,
# andere table nur interval best, vs die anderen
# overlines weg

# nochmal der table oben nur mit clustering solutions, wir sie
# improvement over best CC solution

# try deleting clusters


parse_numbers <- function(x) {
  m <- gregexpr("\\d+", x, perl = TRUE)
  as.numeric(unlist(regmatches(x, m)))
}


politician <- read.csv("bundestag/bundestag/graphs/bundestag_signed_graph_all_periods_id_mapping.csv")
cluster_assignments <- read.csv("heuristics/data/assignments_bundestag.csv", sep=":", nrows = -2, row.names = NULL, header=T)
names(cluster_assignments) <- c("ID","cluster")
cluster_assignments <- cluster_assignments %>%
    filter(row_number() < nrow(cluster_assignments)) %>%
    mutate(ID = as.numeric(ID), cluster = parse_numbers(cluster))

votes <- read.csv("bundestag/all_votes_name_firstname2.csv") %>% arrange(filename)
affilliations <-  unique(votes[c("Bezeichnung","Fraktion.Gruppe","Wahlperiode")])
graph <- read.csv("bundestag/bundestag/graphs/bundestag_signed_graph_all_periods.txt", sep="\t", header = T) %>%
    transmute(from=X..FromNodeId, to=ToNodeId, sign=Sign)

# symmetric closure
sym_graph <- unique(rbind(graph,graph %>% mutate(from2=to,to=from, from=from2,from2=NULL))) %>%
    left_join(cluster_assignments, by =c("to"="ID")) %>%
    group_by(from,cluster) %>%
    summarize(cluster_attraction = sum(sign),edges = sum(abs(sign)))

cluster_assignments_force <- cluster_assignments %>% left_join(sym_graph, by=c("ID"="from")) %>%
    mutate(cluster = cluster.x, cluster.x=NULL) %>%
    group_by(ID,cluster) %>%
    summarise(net_attraction = sum(cluster_attraction*ifelse(cluster.y>cluster+1,1/(cluster.y-cluster),ifelse(cluster.y<cluster,1/(cluster.y-cluster),0)))/2/sum(edges),
              tanh_attraction = tanh(2*net_attraction))


affilliation <- affilliations %>%
    mutate(fraktion = as.factor(parse_parties(Fraktion.Gruppe)), Fraktion.Gruppe = NULL) %>%
    group_by(Bezeichnung,Wahlperiode) %>%
    filter(row_number() == 1) %>%# (takes the initial affiliation of each member)
    left_join(politician, by =c("Bezeichnung"="Person")) %>%
    left_join(cluster_assignments_force) %>%
    mutate(fraktion = factor(fraktion,levels=c("LINKE","GRÜNE","SPD","FDP","CDU/CSU","AFD","fraktionslos","BSW")))

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


tbl <- table(affilliation$cluster,affilliation$fraktion)
d <- dist(t(tbl))
col_order <- seriate(d, method = "OLO")
tbl[ , get_order(col_order) ]

centres    <- 0:7
band_width <- 1.4
stripe_df <- data.frame(
  xmin = seq(-0.5, 6.5, by = 1), # change these widths
  xmax = seq( 0.5, 7.5, by = 1),
  ymin = -Inf,
  ymax =  Inf,
  fill = rep(c("grey95","white"), lengcth.out = length(centres))
)
vals <- (0:7 %*% tbl) / colSums(tbl)



set.seed(5)
ggplot(affilliation %>% filter(fraktion != "fraktionslos"),aes(x=cluster+tanh_attraction*0.8,y=fraktion, color=fraktion, fill=fraktion)) +
     geom_rect(data = stripe_df,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax =  Inf),
            inherit.aes = FALSE, alpha = 0.0,
            fill  = "grey75",        # choose any fill / alpha you like
            colour = "grey75")+geom_violin(alpha=0.3) +
geom_point(shape=21,color="black",position = position_jitternormal(sd_x = 0.02, sd_y = 0.1), alpha=0.5) +
    scale_color_manual(values=c("#BE3075", "#409A3C", "#E3000F", "#FFED00", "#151518", "#009EE0")) +
    scale_fill_manual(values=c("#BE3075", "#409A3C", "#E3000F", "#FFED00", "#151518", "#009EE0")) + theme +
    labs(x="Assigned Cluster",y="Party Affiliation")
striped_bands <- function(centres      = 0:7,     # x positions to centre on
                          band_width   = 1.4,     # total width of each rectangle
                          start_angle  = -45) {   # angle for the first stripe
  data.frame(
    xmin          = centres - band_width/2,
    xmax          = centres + band_width/2,
    ymin          = -Inf,            # run through whole y-range
    ymax          = Inf,
    pattern_angle = rep(c(0,1),
                        length.out = length(centres)/2)   # -45, +45, -45, +45 …
  )
}


ggplot(affilliation %>% filter(fraktion != "fraktionslos"),
       aes(x = cluster + tanh_attraction*0.8,
           y = fraktion,
           color = fraktion,
           fill  = fraktion)) +

  # 3a) booktabs stripes *behind* everything
  geom_rect(
    data        = stripe_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill        = stripe_df$fill,
    colour      = NA
  ) +

  # 3b) your violin + points
  geom_violin(alpha = 0.3) +
  geom_point(
    shape    = 21,
    color    = "black",
    position = position_jitternormal(sd_x = 0.02, sd_y = 0.1),
    alpha    = 0.5
  ) +

  # 3c) your colours *and* custom legend labels
  scale_color_manual(
    values = c(
      LINKE = "#BE3075",
      GRÜNE = "#409A3C",
      SPD   = "#E3000F",
      FDP   = "#FFED00",
      `CDU/CSU` = "#151518",
      AFD   = "#009EE0"
    ),
    labels = c(
      LINKE    = "LINKE (Left Party)",
      GRÜNE    = "GRÜNE (Greens)",
      SPD      = "SPD (Social Democrats)",
      FDP      = "FDP (Free Democrats)",
      `CDU/CSU`= "CDU/CSU (Union)",
      AFD      = "AFD (Alternative)"
    )
  ) +
  scale_fill_manual(
    values = c(
      LINKE = "#BE3075",
      GRÜNE = "#409A3C",
      SPD   = "#E3000F",
      FDP   = "#FFED00",
      `CDU/CSU` = "#151518",
      AFD   = "#009EE0"
    ),
    labels = c(
      LINKE    = "LINKE (Left Party)",
      GRÜNE    = "GRÜNE (Greens)",
      SPD      = "SPD (Social Democrats)",
      FDP      = "FDP (Free Democrats)",
      `CDU/CSU`= "CDU/CSU (Union)",
      AFD      = "AFD (Alternative)"
    )
  ) +

  # 4) finally your labs + theme
  labs(x = "Assigned Cluster", y = "Party Affiliation") +
  theme + theme(panel.grid.minor   = element_blank(),
                panel.grid.major.x = element_blank())


# ---- custom “ticks” data ----
# odd centres get their segment at y = -0.5 and label at -0.4,
# even centres at y = -0.7 and label at -0.8
ticks <- data.frame(
  x0      = sapply(centres - 0.75,function(x)max(-.5,x)),
  x1      = sapply(centres + 0.75,function(x)min(7.5,x)),
  y       = ifelse(centres %% 2 == 1, -0.1, -0.4),
  y_label = ifelse(centres %% 2 == 1, -0.3, -0.2),
  thickness = as.numeric(ifelse(centres %% 2 == 1, 1.2, 2.5)),
  label   = as.character(centres+1)
)

g <- ggplot(affilliation %>% filter(fraktion != "fraktionslos"),
       aes(x = cluster + tanh_attraction*0.8,
           y = fraktion,
           colour = fraktion, fill = fraktion)) +

  # 1) alternating column stripes
  geom_rect(data        = stripe_df,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            inherit.aes = FALSE,
            fill        = stripe_df$fill,
            colour      = NA) +

  # 2) violins & points
  geom_violin(alpha = 0.3) +
  geom_point(shape    = 21,
             colour  = "black",
             position= position_jitternormal(sd_x = 0.02, sd_y = 0.1),
             alpha   = 0.5) +

  # 3) party colours + translated legend
  scale_colour_manual(
    values = c(
      LINKE    = "#BE3075",
      GRÜNE    = "#409A3C",
      SPD      = "#E3000F",
      FDP      = "#FFED00",
      `CDU/CSU`= "#151518",
      AFD      = "#009EE0"
    ),
    labels = c(
      LINKE     = "LINKE (Left)",
      GRÜNE     = "GRÜNE (Green)",
      SPD       = "SPD (Social Democrat)",
      FDP       = "FDP (Liberal)",
      `CDU/CSU` = "CDU/CSU (Conservative)",
      AFD       = "AFD (Right)"
    )
  ) +
  scale_fill_manual(
    values = c(
      LINKE    = "#BE3075",
      GRÜNE    = "#409A3C",
      SPD      = "#E3000F",
      FDP      = "#FFED00",
      `CDU/CSU`= "#151518",
      AFD      = "#009EE0"
    ),
    labels = c(
      LINKE     = "LINKE (Left)",
      GRÜNE     = "GRÜNE (Green)",
      SPD       = "SPD (Social Democrat)",
      FDP       = "FDP (Liberal)",
      `CDU/CSU` = "CDU/CSU (Conservative)",
      AFD       = "AFD (Right)"
    )
  ) +

  # 4) custom segments & labels for x “ticks”
  geom_segment(data = ticks,
               aes(x = x0, xend = x1, y = y, yend = y, linewidth=thickness),
               inherit.aes = FALSE,
               ) +
  geom_text(data = ticks,
            aes(x = (x0 + x1)/2, y = y_label, label = label),
            inherit.aes = FALSE,
            size = 5) + theme + theme(axis.text       = element_blank(),
                                      axis.text.y       = element_blank(),
                                      axis.ticks      = element_blank(),
                                      axis.title = element_blank(),
                                      axis.labels.y = element_blank(),
                                      panel.grid=element_blank(),) + scale_linewidth_identity() +guides(
  colour = guide_legend(nrow = 1, byrow = TRUE),
  fill   = guide_legend(nrow = 1, byrow = TRUE)
) +
  theme(legend.box = "horizontal")


# ---- example plot ----
set.seed(1)
d <- data.frame(x = 0:7, y = rnorm(8))
ggplot(affilliation %>% filter(fraktion != "fraktionslos"),aes(x=cluster+rnorm(length(cluster))*0.2,y=fraktion, color=fraktion, fill=fraktion)) +
    geom_rect_pattern(
    data            = striped_bands(),            # <- our helper
    aes(xmin = xmin, xmax = xmax,
        ymin = ymin, ymax = ymax,
        ),
    pattern_angle = rep(c(45,-45),4),
    pattern         = "stripe",
    pattern_fill    = "white",    # colour of the stripes
    pattern_colour  = "grey50",    # outline of stripes
    pattern_density = 0.45,        # how much of the area is inked
    pattern_spacing = 0.03,        # distance between stripes (npc units)
    inherit.aes     = FALSE,        # rectangles ignore the main aes mapping
    pattern_alpha = .3,
    alpha = .3
  ) + geom_violin(alpha=0.6) +
geom_point(shape=21,color="black",position = position_jitternormal(sd_x = 0.0, sd_y = 0.08), alpha=0.5) +
    scale_color_manual(values=c("#BE3075", "#409A3C", "#E3000F", "#FFED00", "#151518", "#009EE0")) +
    scale_fill_manual(values=c("#BE3075", "#409A3C", "#E3000F", "#FFED00", "#151518", "#009EE0")) + theme



# stat_summary(
#     geom    = "errorbarh",              # horizontal bar
#     fun     = mean,                     # mid-point
#     fun.min = \(z) quantile(z+rnorm(length(z))*0.5,.25),     # left end
#     fun.max = \(z) quantile(z+rnorm(length(z))*0.5,.75),     # right end
#     height  = 0,                      # vertical thickness of the bar
#     linewidth = 35,
#     alpha=.7
#   ) +


g <- ggplot(affilliation %>% filter(fraktion != "fraktionslos"),aes(x=cluster, color=fraktion, fill=fraktion)) +
geom_bar(shape=21,color="black", alpha=0.6) +
    scale_color_manual(values=c("#BE3075", "#409A3C", "#E3000F", "#FFED00", "#151518", "#009EE0")) +
    scale_fill_manual(values=c("#BE3075", "#409A3C", "#E3000F", "#FFED00", "#151518", "#009EE0")) + theme + scale_linewidth_identity() +guides(
  colour = guide_legend(nrow = 1, byrow = TRUE),
  fill   = guide_legend(nrow = 1, byrow = TRUE)
) +
  theme(legend.box = "horizontal") + scale_x_continuous(
  breaks     = 0:7,
  labels     = 1:8,      # hide the default numbers
) +
  theme(
    axis.ticks.x = element_line(),   # turn on tick‐marks
  ) + scale_fill_manual(
  values = c(
    LINKE    = "#BE3075",
    GRÜNE    = "#409A3C",
    SPD      = "#E3000F",
    FDP      = "#FFED00",
    `CDU/CSU`= "#151518",
    AFD      = "#009EE0"
  ),
  labels = c(
    LINKE     = "LINKE (Left)",
    GRÜNE     = "GRÜNE (Green)",
    SPD       = "SPD (Social Democrat)",
    FDP       = "FDP (Liberal)",
    `CDU/CSU` = "CDU/CSU (Conservative)",
    AFD       = "AFD (Right)"
  )
) + labs(y = "Number of Assigned Members", x = "Interval Index")



