# Bitcoin Price Prediction 📈💰

## 📌 Project Overview
Bitcoin is one of the most volatile assets in the financial market, making its price prediction a challenging task. This project aims to predict Bitcoin prices using historical market data and machine learning models. Through data preprocessing, exploratory data analysis (EDA), feature engineering, and model evaluation, we build a predictive model that provides insights into Bitcoin's price trends.

## 📂 Dataset Used
- **bitcoin_dataset.csv** (Training Dataset)
- **test_set.csv** (Testing Dataset)

### **Dataset Features:**
The dataset contains multiple attributes related to Bitcoin market metrics, including:
- `Date` – Date of observation  
- `btc_market_price` – Average USD market price across major Bitcoin exchanges  
- `btc_total_bitcoins` – Total number of mined Bitcoins  
- `btc_market_cap` – Total market capitalization  
- `btc_trade_volume` – USD trading volume on major exchanges  
- `btc_blocks_size` – Total size of all block headers and transactions  
- `btc_avg_block_size` – Average block size in MB  
- `btc_n_orphaned_blocks` – Number of blocks mined but not attached to the main blockchain  
- `btc_n_transactions_per_block` – Average transactions per block  
- `btc_median_confirmation_time` – Median transaction confirmation time  
- `btc_hash_rate` – Estimated network hash rate (TH/s)  
- `btc_difficulty` – Mining difficulty level  
- `btc_miners_revenue` – Revenue earned by miners  
- `btc_transaction_fees` – Total transaction fees paid to miners  
- `btc_cost_per_transaction_percent` – Miners' revenue as a percentage of transaction volume  
- `btc_cost_per_transaction` – Miners' revenue per transaction  
- `btc_n_unique_addresses` – Total unique addresses used  
- `btc_n_transactions` – Daily confirmed transactions  
- `btc_n_transactions_total` – Total number of transactions  
- `btc_n_transactions_excluding_popular` – Transactions excluding the top 100 popular addresses  
- `btc_n_transactions_excluding_chains_longer_than_100` – Transactions per day excluding long transaction chains  
- `btc_output_volume` – Total value of all transaction outputs per day  
- `btc_estimated_transaction_volume` – Estimated transaction volume  

## 🛠️ Technologies & Libraries Used
- **Python** – NumPy, Pandas, Matplotlib, Seaborn, Plotly  
- **Machine Learning** – Scikit-Learn  
- **Data Visualization** – Matplotlib, Seaborn, Plotly  
- **Jupyter Notebook** for development  

## 🔍 Data Preprocessing
- Handled missing data effectively  
- Applied **Min-Max Scaling** for feature normalization  
- Used **Cross-Validation** for model training and evaluation  

## 📊 Machine Learning Models Used
To achieve the best prediction, multiple models were tested, including:  
- **K-Neighbors Regression**  
- **Linear Regression**  
- **Ridge Regression**  
- **Lasso Regression**  
- **Polynomial Regression**  
- **Support Vector Machine (SVM)**  

After evaluating all models, **Polynomial Lasso Regression** was found to provide the best performance with the lowest error.  

## 🚀 Results & Predictions
- **Polynomial Lasso Regression** provided the most accurate predictions.  
- Feature importance analysis highlighted key attributes influencing Bitcoin price fluctuations.  
- The model was validated using cross-validation techniques.  

## 🏗️ How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/bitcoin-price-prediction.git

## 🔮 Future Enhancements
- Implementing deep learning techniques like LSTMs with fine-tuned hyperparameters
- Incorporating external macroeconomic indicators (e.g., stock market trends, interest rates)
- Enhancing feature engineering for better model accuracy
- Deploying the model using Flask or Streamlit for real-time predictions   

   
