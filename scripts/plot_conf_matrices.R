library(ggplot2)
library(caret)
library(dplyr)
library(ggthemes)
library(stringr)
library(gridExtra)

plot_cm <- function(fname, tit) { 
  df <- read.csv(fname)
  
  table <- data.frame(confusionMatrix(df$class_pred, df$class_label)$table)
  
  table$Reference <- as.factor(str_replace_all(table$Reference, "\\Q|\\E", " "))
  table$Prediction <- as.factor(str_replace_all(table$Prediction, "\\Q|\\E", " "))
  
  plotTable <- table %>%
    mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
    group_by(Reference) %>%
    mutate(prop = Freq/sum(Freq))
  
  # fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups (see dplyr code above as well as original confusion matrix for comparison)
  ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
    geom_tile() +
    geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
    scale_fill_manual(values = c(good = "green", bad = "red")) +
    theme_few() +
    xlim(rev(levels(table$Reference))) +
    ggtitle(tit) +
    theme(
      axis.text.x = element_text(angle = 30, hjust=1),
      legend.position = "none",
      plot.title = element_text()
    )
}
  
svevo_ctx <- plot_cm('generated_text_labels=contextual_samples=3_predicted.csv', tit="Svevo CTX")
svevo_ctx_ht <- plot_cm('generated_text_labels=contextual_samples=3_high_temp_predicted.csv', tit="Svevo CTX High Temp.")
eur_it <- plot_cm('generated_text_labels_europarl_it=contextual_samples=3_predicted_restricted.csv', tit="Europarl-IT CTX")
eur_it_ht <- plot_cm('generated_text_labels_europarl_it=contextual_samples=3_high_temp_predicted.csv', tit="Europarl-IT CTX High Temp.")
eur_en <- plot_cm('generated_text_labels_europarl_en=contextual_samples=3_predicted_restricted.csv', tit="Europarl-EN CTX")
eur_en_ht <- plot_cm('generated_text_labels_europarl_en=contextual_samples=3_high_temp_predicted.csv', tit="Europarl-EN CTX High Temp.")
grid.arrange(svevo_ctx, svevo_ctx_ht, nrow=1)
grid.arrange(eur_it, eur_it_ht, nrow=1)
grid.arrange(eur_en, eur_en_ht, nrow=1)

svevo_tuned <- plot_cm('svevo_generated_tuned_predicted.csv', tit="Svevo Tuned")
grid.arrange(svevo_ctx, svevo_tuned, nrow=1)
eur_en_tuned <- plot_cm('generated_EuroParlEng_contextual_samp=1_iters=20_temp=1_0_gm=0_95.csv', tit="Europarl-EN CTX Tuned")
grid.arrange(eur_en, eur_en_tuned, nrow=1)
eur_it_tuned <- plot_cm('generated_EuroParlIta_contextual_samp=1_iters=20_temp=1_0_gm=0_95.csv', tit="Europarl-IT CTX Tuned")
grid.arrange(eur_it, eur_it_tuned, nrow=1)


  # Plot topic modeling results

data <- read.csv('topic_modeling_results.csv') %>% filter(dataset %in% c("Svevo Corpus"))
data$dataset <- factor(data$dataset, levels=c("EuroParl EN", "EuroParl IT"))
data$type <- factor(data$type, levels=c("combined", "contextual"))
number_ticks <- function(n) {function(limits) pretty(limits, n)}
palette = c("#D85885", "#7694C1")
ggplot(data, aes(x=type, y=npmi, fill=type)) + 
  geom_col() +
  facet_grid(~ n_topics, scales="free_y") +
  scale_y_continuous(breaks=scales::pretty_breaks(3)) +
  scale_fill_manual(values = palette) + 
  ylab("NPMI") +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.y = element_text(vjust=2),
        legend.direction = "horizontal",
        legend.title = element_blank(),
        legend.position = "bottom",
        legend.text = element_text(margin = margin(r = 10, unit = "pt")),
        strip.text.y = element_text(face="bold"))
ggsave("sentence_scores.png", units="mm", width=183, height=118, dpi=300)
