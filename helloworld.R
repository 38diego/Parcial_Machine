.libPaths("/home/vscode/R_packages")
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", lib = "/home/vscode/R_packages")
}
library(ggplot2, lib.loc = "/home/vscode/R_packages")

# Generar un conjunto de datos de ejemplo
set.seed(123)
n <- 100
x <- rnorm(n, mean = 5, sd = 2)
y <- 3 + 1.5 * x + rnorm(n, sd = 1)
weights <- runif(n, min = 0.5, max = 1.5)  # Pesos aleatorios


modelo_sin_ponderacion <- lm(y ~ x)

modelo_con_ponderacion <- lm(y ~ x, weights = weights)

datos <- data.frame(x = x, y = y)
datos$pred_sin_ponderacion <- predict(modelo_sin_ponderacion)
datos$pred_con_ponderacion <- predict(modelo_con_ponderacion)

# Graficar los datos y las dos líneas de regresión
ggplot(datos, aes(x = x, y = y)) +
  geom_point(aes(size = weights), alpha = 0.5) +  # Tamaño de puntos según los pesos
  geom_line(aes(y = pred_sin_ponderacion), color = "blue", size = 1, linetype = "dashed") +
  geom_line(aes(y = pred_con_ponderacion), color = "red", size = 1) +
  labs(
    title = "Comparación de Regresión: Sin Ponderación vs. Con Ponderación",
    x = "Variable Independiente (x)",
    y = "Variable Dependiente (y)"
  ) +
  scale_size_continuous(name = "Pesos") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(size = guide_legend(title = "Pesos")) +
  annotate("text", x = min(x), y = max(y), label = "Sin Ponderación", color = "blue", hjust = 0, vjust = 1) +
  annotate("text", x = min(x), y = max(y) - 1, label = "Con Ponderación", color = "red", hjust = 0, vjust = 1)

  ggsave('plot.png')