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
