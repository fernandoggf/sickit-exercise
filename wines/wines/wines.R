
library(readr)
library(ggplot2)
library(lattice)
library(lubridate)
library(dplyr)
require(stats)
library(e1071)
library(nlme)

# Carga de datos en esquema de data frame
data <- read_csv("/Users/fernandofigueroa/Documents/R/wines/winequality-red.csv")
wine <- as.data.frame(data)
wine

man1 <- manova(cbind(wine$`fixed acidity`,wine$`volatile acidity`,wine$`citric acid`, 
                     wine$`residual sugar`,wine$chlorides,wine$`free sulfur dioxide`,wine$`total sulfur dioxide`,
                     wine$density, wine$pH,wine$sulphates,wine$alcohol)~wine$quality, data=wine)
summary(man1)

model2 <- lm(wine$quality ~ ., data = wine)
summary(model2)
