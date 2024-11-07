.libPaths("/home/vscode/R_packages")
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", lib = "/home/vscode/R_packages")
}
library(ggplot2, lib.loc = "/home/vscode/R_packages")

if (!requireNamespace("glmnet", quietly = TRUE)) {
  install.packages("glmnet", lib = "/home/vscode/R_packages")
}

library(glmnet, lib.loc = "/home/vscode/R_packages")

# Generamos algunos datos de ejemplo
set.seed(123)
n <- 100
x <- rnorm(n)
y <- 3 * x + rnorm(n)

# Preparar los datos para glmnet (convertimos a matriz)
X <- as.matrix(cbind(1, x))  # Matriz de predictores (con columna de 1s para la intersección)

# Ajuste con OLS (usamos la fórmula directa)
ols_model <- lm(y ~ x)
summary(ols_model)

# Ajuste con glmnet para Ridge (regularización L2)
ridge_model <- glmnet(X, y, alpha = 0)  # alpha = 0 para Ridge

# Ajuste con glmnet para Lasso (regularización L1)
lasso_model <- glmnet(X, y, alpha = 1)  # alpha = 1 para Lasso

# Extraemos los coeficientes para lambda = 0 (sin regularización)
coef(ridge_model, s = 0)
coef(lasso_model, s = 0)

# Graficamos los resultados
df <- data.frame(x = x, y = y)

# Predicciones con OLS, Ridge y Lasso
pred_ols <- predict(ols_model, newdata = data.frame(x = x))
pred_ridge <- predict(ridge_model, s = 0, newx = X)  # s = 0 para el valor de lambda = 0
pred_lasso <- predict(lasso_model, s = 0, newx = X)  # s = 0 para el valor de lambda = 0

# Graficamos los resultados con leyenda
ggplot(df, aes(x = x, y = y)) +
  geom_point(aes(color = "Datos reales"), alpha = 0.7) +
  geom_line(aes(y = pred_ols, color = "Ajuste OLS"), linetype = "dashed",size = 1) +
  geom_line(aes(y = pred_ridge, color = "Ajuste Ridge (glmnet)"), linetype = "solid",size = 1) +
  geom_line(aes(y = pred_lasso, color = "Ajuste Lasso (glmnet)"), linetype = "dotdash",size = 1) +
  labs(title = "Comparación de ajuste: OLS, Ridge y Lasso",
       color = "Líneas") +
  theme_minimal() +
  scale_color_manual(values = c("Datos reales" = "gray", 
                                "Ajuste OLS" = "blue", 
                                "Ajuste Ridge (glmnet)" = "red", 
                                "Ajuste Lasso (glmnet)" = "green"))

ggsave('glmnet.png')