# Library
library(dplyr)
library(factoextra)
library(caret)
library(corrplot)

# DATASET 1: CITY_LIFESTYLE =====
# Model : Clustering (K-Means)
# Goal: Segment cities into distinct lifestyle groups

# Set directory
# Load dataset
city_data <- read.csv("city_lifestyle_dataset.csv")

# Summary and data check
summary(city_data)
cat("\nMissing values per column:\n")
print(colSums(is.na(city_data)))

# Data Preparation (select features)
cluster_data_city <- city_data %>%
  select(population_density, avg_income, avg_rent, public_transport_score, green_space_ratio)

# Scaling
scaled_data <- scale(cluster_data_city)

# Determine optimal k (Elbow method)
fviz_nbclust(scaled_data, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2) +
  labs(title = "Elbow Method for Optimal Number of Clusters",
       subtitle = "Dashed line indicates selected k = 3")

# Apply K-Means with k = 3
set.seed(123)
kmeans_result <- kmeans(scaled_data, centers = 3, nstart = 25)

cat("\n--- K-means model output ---\n")
print(kmeans_result)

# Attach cluster membership to original data
city_data$cluster <- as.factor(kmeans_result$cluster)

# Create readable cluster labels (for interpretation only)
cluster_labels <- c(
  "1" = "High-rent, high-income cities",
  "2" = "Moderate income, greener cities",
  "3" = "Lower-rent, lower-density cities"
)

city_data$cluster_name <- cluster_labels[as.character(city_data$cluster)]

cat("\n--- Cluster label mapping used for interpretation ---\n")
print(cluster_labels)

# Visualize clusters (PCA projection used by factoextra)
fviz_cluster(kmeans_result, data = scaled_data,
             palette = c("lightblue", "lightgreen", "lightpink"),
             geom = "point",
             ellipse.type = "convex",
             ggtheme = theme_bw(),
             main = "City Lifestyle Clusters")

# Cluster interpretation table (means in original units)
cluster_summary <- city_data %>%
  group_by(cluster, cluster_name) %>%
  summarise(
    Avg_Population_Density = mean(population_density, na.rm = TRUE),
    Avg_Income = mean(avg_income, na.rm = TRUE),
    Avg_Rent = mean(avg_rent, na.rm = TRUE),
    Avg_Public_Transport = mean(public_transport_score, na.rm = TRUE),
    Avg_Green_Space = mean(green_space_ratio, na.rm = TRUE),
    Count = n(),
    .groups = "drop"
  )

cat("\n--- Cluster Characteristics (Original Scale) ---\n")
print(cluster_summary)

# DATASET 2: REAL ESTATE =====
# Model: Multiple Linear Regression
# Goal: Predict Price using housing characteristics

# Load data
real_estate <- read.csv("real_estate_dataset.csv", stringsAsFactors = FALSE)

# Summary and data check
summary(real_estate)
cat("\nMissing values per column:\n")
print(colSums(is.na(real_estate)))

# Data cleaning/ preparation
re_clean <- real_estate %>% select(-ID)

# 2) Convert binary columns to factor (important for lm + new_house prediction)
if ("Has_Garden" %in% names(re_clean)) {
  re_clean$Has_Garden <- factor(re_clean$Has_Garden, levels = c(0, 1), labels = c("No", "Yes"))
}
if ("Has_Pool" %in% names(re_clean)) {
  re_clean$Has_Pool <- factor(re_clean$Has_Pool, levels = c(0, 1), labels = c("No", "Yes"))
}

# Remove missing values
re_clean <- na.omit(re_clean)

# Exploratory Data Analysis (EDA)
# Distribution of property prices (Histogram)
hist(re_clean$Price,
     breaks = 20,
     main = "Distribution of Property Prices",
     xlab = "Price")

# Correlation Plot
cols_numeric <- real_estate %>%
  select(Square_Feet,
         Num_Bedrooms,
         Num_Bathrooms,
         Num_Floors,
         Year_Built,
         Garage_Size,
         Location_Score,
         Distance_to_Center,
         Price)

# Calculate correlation matrix
cor_matrix <- cor(cols_numeric, use = "complete.obs")
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         col = colorRampPalette(c("#B2182B", "white", "#2166AC"))(200), # clearer contrast
         tl.col = "black",
         tl.cex = 1.1,        # ⬅ increase variable label size
         tl.srt = 30,         # ⬅ less rotation
         addCoef.col = "black",
         number.cex = 0.9,    # ⬅ bigger correlation numbers
         cl.cex = 0.9,        # color legend size
         mar = c(0, 0, 2, 0))

title(main = "Correlation Heatmap of Real Estate Features", adj = 0)

# EDA: Scatter Plot Matrix
re_reg_vars <- re_clean %>%
  select(Price, Square_Feet, Num_Bedrooms, Year_Built)

splom(~re_reg_vars,
      data = re_reg_vars,
      axis.line.tck = 0,
      axis.text.alpha = 0.7)

# Train/Test Split
set.seed(123)
index_re <- createDataPartition(re_clean$Price, p = 0.8, list = FALSE)
train_re <- re_clean[index_re, ]
test_re  <- re_clean[-index_re, ]

# Build Model
model_price <- lm(Price ~ ., data = train_re)

# Summary
summary(model_price)

# Evaluation (RMSE on test set)
pred_price <- predict(model_price, newdata = test_re)
rmse_re <- sqrt(mean((test_re$Price - pred_price)^2))
cat("\nHouse Price RMSE (Test Set):", round(rmse_re, 2), "\n")

# Visualization: Actual vs Predicted
plot(test_re$Price, pred_price,
     pch = 19,
     col = "blue",
     xlab = "Actual Price",
     ylab = "Predicted Price",
     main = "Real Estate: Actual vs Predicted Price")

abline(a = 0, b = 1, col = "red", lwd = 2)

# Prediction for a New House
new_house <- data.frame(
  Square_Feet = 150,
  Num_Bedrooms = 3,
  Num_Bathrooms = 2,
  Num_Floors = 2,
  Year_Built = 2005,
  Has_Garden = factor("Yes", levels = levels(train_re$Has_Garden)),
  Has_Pool   = factor("No",  levels = levels(train_re$Has_Pool)),
  Garage_Size = 30,
  Location_Score = 7.5,
  Distance_to_Center = 10
)

pred_new_house <- predict(model_price, newdata = new_house)
cat("\nPredicted Price for new_house:", round(pred_new_house, 2), "\n")

