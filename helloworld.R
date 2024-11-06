.libPaths("/home/vscode/R_packages")
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", lib = "/home/vscode/R_packages")
}
library(ggplot2, lib.loc = "/home/vscode/R_packages")

set.seed(123)
n <- 100
x <- rnorm(n, mean = 5, sd = 2)
y <- 3 + 1.5 * x + rnorm(n, sd = 1)

# Asignar pesos más extremos
weights <- ifelse(x > 5, 100, 0.01)  # Pesos muy altos para x > 5, muy bajos para el resto

# Ajustar el modelo sin ponderación
modelo_sin_ponderacion <- lm(y ~ x)

# Ajustar el modelo con ponderación
modelo_con_ponderacion <- lm(y ~ x, weights = weights)

# Crear un data frame con las predicciones de ambos modelos
datos <- data.frame(x = x, y = y, weights = weights)
datos$pred_sin_ponderacion <- predict(modelo_sin_ponderacion)
datos$pred_con_ponderacion <- predict(modelo_con_ponderacion)

# Crear una nueva variable para resaltar los puntos con pesos altos
datos$highlight <- ifelse(datos$weights == 100, "Peso Alto", "Peso Bajo")

# Graficar los datos y las dos líneas de regresión
ggplot(datos, aes(x = x, y = y)) +
  geom_point(aes(color = highlight), alpha = 0.7) +  # Colorear según el tipo de peso
  geom_line(aes(y = pred_sin_ponderacion, color = "Sin Ponderación"), size = 1, linetype = "dashed") +
  geom_line(aes(y = pred_con_ponderacion, color = "Con Ponderación"), size = 1) +
  labs(
    title = "Comparación de Regresión: Sin Ponderación vs. Con Ponderación",
    x = "Variable Independiente (x)",
    y = "Variable Dependiente (y)",
    color = "Modelo"
  ) +
  scale_color_manual(
    name = "",
    values = c("Sin Ponderación" = "blue", "Con Ponderación" = "red", "Peso Alto" = "red", "Peso Bajo" = "gray")
  ) +  # Colorear las líneas y los puntos
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(size = guide_legend(title = "Pesos"))

ggsave('plot.png')