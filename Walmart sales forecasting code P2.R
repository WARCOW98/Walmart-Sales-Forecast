## ----------------------------------------------------------
## LOAD LIBRARIES
## ----------------------------------------------------------
# Load required libraries for time series analytics.
library(forecast)
library(zoo)         
library(dplyr)       
library(lubridate)   
library(ggplot2)
library(tseries)

## ----------------------------------------------------------
## LOAD DATA
## ----------------------------------------------------------

# Set working directory (set according to your working directory)
setwd("/Users/sujanmanohar/Desktop/Time Series")

# Load Data
project_df <- read.csv("Walmart Qtr.csv",, header = TRUE, stringsAsFactors = FALSE)  # Replace with actual path
project_df


## ----------------------------------------------------------
## EXPLORE & VISUALIZE THE SERIES
## ----------------------------------------------------------
# Convert the Date column to Date format and sort the data chronologically.
project_df$Date <- as.Date(project_df$Date, format = "%m/%d/%y")

project_df$Quarter <- paste0(year(project_df$Date), " Q", quarter(project_df$Date))

#Rename Coloumns
project_df <- project_df %>% rename(
  Date = `Date`,
  Revenue = `Revenue.Millions.of.US...`
)

# Remove commas from Revenue and convert to numeric
project_df$Revenue <- as.numeric(gsub(",", "", project_df$Revenue))
project_df

project_df <- project_df %>% arrange(Date)
project_df

project_df <- project_df %>% select(Quarter, Revenue)
head(project_df)
tail(project_df)

#Summary of dataset
summary(project_df)

project_df$Quarter <- factor(project_df$Quarter, levels = unique(project_df$Quarter))
project_df

#Apply the ggplot() function to plot the historical data.
ggplot(project_df, aes(x = Quarter, y = Revenue, group = 1)) +
  geom_line(color = "blue", linewidth = 1) +  
  labs(title = "Walmart Quarterly Revenue", x = "Quarter", y = "Revenue (in Millions)") +
  scale_x_discrete(breaks = project_df$Quarter[seq(1, length(project_df$Quarter), by = 4)]) +  # Show label every 4 quarters
  theme_minimal() +  # Apply a clean theme
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10),  # Rotate x-axis labels
    panel.grid.major = element_line(color = "black", size = 0.3),  # Black major grid lines
    panel.grid.minor = element_line(color = "black", size = 0.2),  # Black minor grid lines
    panel.background = element_blank()  # Remove background for cleaner look
  )


## ----------------------------------------------------------
## ----------Create the Quarterly time series ---------------
walmart.ts <- ts(project_df$Revenue, 
                 start = c(2009, 1), end = c(2024, 4), freq = 4)
walmart.ts

## ----------------------------------------------------------
## ----------PARTITION THE TIME SERIES-----------------------
# Partition the quarterly series into a training set and a validation set.

nValid <- 16
length(walmart.ts)
nTrain <- length(walmart.ts) - nValid
train.ts <- window(walmart.ts, start = c(2009, 1), end = c(2009, nTrain))
train.ts
valid.ts <- window(walmart.ts, start = c(2009, nTrain + 1), 
                   end = c(2009, nTrain + nValid))
valid.ts

# Plot the autocorrelation function for the Quarterly series (lags up to 12).
Acf(walmart.ts, lag.max = 12, main = "Autocorrelation for Quaterly Revenue for Walmart")


# Plot the full series with partition annotations
plot(walmart.ts, xlab = "Year", ylab = "Revenue (in Millions)",
     main = "Quarterly Revenue with Training and Validation Partitions", lwd = 2, xaxt = "n")

# Generate sequence of years for the x-axis
years_seq <- seq(2009, 2024, by = 1)
axis(1, at = years_seq, labels = years_seq)

# Overlay training partition
lines(train.ts, col = "blue", lwd = 2)

# Add a vertical dashed line at the boundary between training and validation
abline(v = time(train.ts)[length(train.ts)], col = "red", lwd = 2, lty = 2)

# Add text labels for Training and Validation portions
legend("topleft",
       legend = c("Training", "Validation"),
       col    = c("blue", "black"),
       lty    = c(1, 1),
       lwd    = c(2, 2),
       bty    = "n")


# Decompose the series using STL to show level, trend, seasonality, and remainder.
walmart.stl <- stl(walmart.ts, s.window = "periodic")
autoplot(walmart.stl, main = "STL Decomposition of Quarterly Walmart Revenue")

## ----------------------------------------------------------
## ADF Test and Differencing

adf.test(walmart.ts)

diff_walmart <- diff(walmart.ts, differences = 1)
adf.test(diff_walmart) # p-value >0.05

diff_seasonal_walmart <- diff(diff_walmart, lag = 4)  # Seasonal differencing
adf.test(diff_seasonal_walmart)

Acf(diff_seasonal_walmart, lag.max = 12, main = "Autocorrelation after differencing and Seasonal differencing")

## ----------------------------------------------------------
## APPLY FORECASTING METHODS ON TRAINING SET & VALIDATE

## ----------------------------------------------------------
## AUTO ARIMA MODEL

### (a) AUTO ARIMA MODEL
# Fit an ARIMA model on the training set using auto.arima().
arima.model <- auto.arima(train.ts)
summary(arima.model)

checkresiduals(arima.model)

# Apply forecast() function to make predictions for the validation set.
train.arima.pred <- forecast(arima.model, h = nValid, level = 0)
train.arima.pred

# Use Acf() function to create autocorrelation chart of ARIMA model residuals in training.
Acf(arima.model$residuals, lag.max = 12,
    main = "Autocorrelations of ARIMA Model Residuals in Training Period")

# Plot time series data, ARIMA model, and predictions for validation period.
plot(train.arima.pred,
     xlab = "Time", ylab = "Revenue (in Millions)", ylim = c(95000, 180000),
     bty = "l", xlim = c(2009, 2025), xaxt = "n",
     main = "ARIMA Model Forecast for Walmart Revenue", lwd = 2, flty = 5)

# Generate sequence of years for x-axis
years_seq <- seq(2009, 2025, by = 1)
axis(1, at = years_seq, labels = years_seq)

# Overlay the validation data in black.
lines(valid.ts, col = "black", lwd = 2, lty = 1)

# Add a vertical dashed line at the boundary between training and validation
train_end <- end(train.ts)  # Get the last time index of training set
abline(v = train_end[1] + (train_end[2] - 1) / 4, col = "red", lwd = 2, lty = 2)

# Create a legend.
legend("topleft",
       legend = c("Revenue Time Series",
                  "ARIMA Forecast for Validation Period"),
       col    = c("black", "blue"),
       lty    = c(1, 1, 5), lwd = c(2, 2, 2), bty = "n")


## ----------------------------------------------------------
## -----------HOLT-WINTER'S AUTOMATIC (ZZZ) MODEL------------
## ----------------------------------------------------------

### (b) HOLT-WINTER'S AUTOMATIC PARAMETER SELECTION MODEL
# Fit a Holt-Winters model using automatic parameter selection (model = "ZZZ").
hw.model <- ets(train.ts, model = "ZZZ")
summary(hw.model)

# Forecast the validation set.
train.hw.pred <- forecast(hw.model, h = nValid, level = 0)
train.hw.pred

# Acf() for Holt-Winters residuals.
Acf(hw.model$residuals, lag.max = 12,
    main = "Autocorrelations of Holt-Winters Model Residuals in Training Period")

# Plot Holt-Winters forecast similarly.
plot(train.hw.pred,
     xlab = "Time", ylab = "Revenue (in Millions)", ylim = c(95000, 180000),
     bty = "l", xlim = c(2009, 2026), xaxt = "n",
     main = "HW Model Forecast for Walmart Revenue", lwd = 2, flty = 5)
axis(1, at = seq(2009, 2026, 1), labels = format(seq(2009, 2026, 1)))


lines(valid.ts, col = "black", lwd = 2, lty = 1)

abline(v = time(train.ts)[length(train.ts)], col = "red", lwd = 2, lty = 2)

legend("topleft",
       legend = c("Revenue Time Series",
                                "HW Forecast for Validation Period"),
       col = c("black", "blue"),
       lty = c(1, 1, 5), lwd = c(2, 2, 2), bty = "n")


## ----------------------------------------------------------
## EVALUATE & COMPARE MODEL PERFORMANCE (Validation)
## ----------------------------------------------------------
accuracy_arima <- round(accuracy(train.arima.pred$mean, valid.ts), 3)
accuracy_hw   <- round(accuracy(train.hw.pred$mean, valid.ts), 3)
print("Validation Accuracy Measures:")
print("ARIMA:")
print(accuracy_arima)
print("HW:")
print(accuracy_hw)
print("HW:")

## ----------------------------------------------------------
## FINAL MODELING ON THE ENTIRE DATA SET
## ----------------------------------------------------------
# Fit final models on the entire monthly series.
final_arima <- auto.arima(walmart.ts)

# Holt-Winters with automatic parameter selection (model = "ZZZ")
final_hw    <- ets(walmart.ts, model = "ZZZ")

summary(final_arima)

summary(final_hw)

# Forecast into the future for 16 periods.
final_arima_pred <- forecast(final_arima, h = 16, level =0)
final_hw_pred    <- forecast(final_hw, h = 16, level = 0)



## ----------------------------------------------------------
## COMPARE ACCURACY OF FINAL MODELS
## ----------------------------------------------------------
# Calculate accuracy measures on the entire series fitted values.
acc_final_arima <- round(accuracy(final_arima_pred$fitted, walmart.ts), 3)
acc_final_hw    <- round(accuracy(final_hw_pred$fitted, walmart.ts), 3)
acc_final_naive <- round(accuracy(naive(walmart.ts)$fitted, walmart.ts), 3)
acc_final_snaive<- round(accuracy(snaive(walmart.ts)$fitted, walmart.ts), 3)

print("Final Model Accuracy Measures (Entire Series):")
print("ARIMA:")
print(acc_final_arima)
print("Holt-Winters (ZZZ):")
print(acc_final_hw)
print("Naïve:")
print(acc_final_naive)
print("Seasonal Naïve:")
print(acc_final_snaive)

# Create a summary table for RMSE and MAPE.
final_accuracy <- data.frame(
  Model = c("ARIMA", "Holt-Winters", "Naïve", "Seasonal Naïve"),
  RMSE = c(acc_final_arima["Test set","RMSE"],
           acc_final_hw["Test set","RMSE"],
           acc_final_naive["Test set","RMSE"],
           acc_final_snaive["Test set","RMSE"]),
  MAPE = c(acc_final_arima["Test set","MAPE"],
            acc_final_hw["Test set","MAPE"],
           acc_final_naive["Test set","MAPE"],
           acc_final_snaive["Test set","MAPE"])
)
print("Summary of Final Model Accuracy Measures:")
print(final_accuracy)

# Identify the best model based on the lowest RMSE and MAPE.
best_rmse_model <- final_accuracy$Model[which.min(final_accuracy$RMSE)]
best_mape_model <- final_accuracy$Model[which.min(final_accuracy$MAPE)]
cat("The best model based on RMSE is:", best_rmse_model, "\n")
cat("The best model based on MAPE is:", best_mape_model, "\n")

## ----------------------------------------------------------
## 11. IMPLEMENT FINAL FORECAST INTO THE FUTURE
## ----------------------------------------------------------
# Using the best model, use it to forecast.

# Using the best model (final_hw)
final_model <- final_hw
summary(final_hw)
future_period <- 16
final_forecast <- forecast(final_model, h = future_period, level = 0)
final_forecast

# Plot the final forecast with base R
plot(final_forecast,
     xlab = "Time", ylab = "Revenue",
     xlim = c(2009, 2030),     
     ylim = c(80000, 220000),         
     main = "Forecasted Walmart Revenue for the Next 16 Quarters",
     lwd = 2, flty = 1)     

# Add a legend explaining the lines
legend("topleft",
       legend = c("Historical Series",
                  "Forecast (Next 16 Quarters)"),
       col = c("black", "blue"),
       lty = c(1, 1, 5),
       lwd = c(2, 2, 2),
       bty = "n")






















