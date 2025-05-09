# Dynamic Pricing and Sales Analysis System 🛒📈

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dynamic-pricing-x2.streamlit.app/) <!-- Replace with your deployed Streamlit app URL -->

This project analyzes retail transaction data to understand hourly sales trends and implements a dynamic pricing simulation based on demand fluctuations during operating hours. It includes features like item performance categorization, market basket analysis, profitability simulation, and an interactive Streamlit web application for visualization and simulation.

## Features

*   **Data Loading & Processing:** Reads transaction data from an Excel file (`dma_data.xlsx`), cleans it, and structures it for analysis. Handles potential data inconsistencies.
*   **Hourly Sales Analysis:**
    *   Visualizes hourly sales quantity trends for top-selling items.
    *   Displays an overall hourly sales trend for all items combined.
    *   Presents a heatmap showing peak sales hours for top items.
*   **Revenue & Profit Analysis:**
    *   Calculates and visualizes hourly revenue and *estimated* profit trends (based on an assumed cost factor).
    *   Shows total revenue and profit generated by top items.
*   **Market Basket Analysis:**
    *   Uses the Apriori algorithm (`mlxtend`) to find frequently co-purchased items.
    *   Displays frequent itemsets and association rules (e.g., "If Milk is bought, then Cookies are bought X% of the time").
    *   Allows adjusting minimum support and confidence thresholds interactively.
*   **Item Performance Categorization:**
    *   Classifies items based on sales volume (total quantity) and sales consistency (standard deviation during open hours) into categories like "Star," "Cash Cow," "Question Mark," etc. (simplified categories used in the app).
    *   Visualizes categories on a scatter plot.
*   **Dynamic Pricing Engine:**
    *   Calculates a base price (median) for each item.
    *   Determines hourly demand deviation using Z-scores relative to the average demand *during shop operating hours*.
    *   Adjusts prices within user-defined limits based on demand (higher price for higher demand, lower price for lower demand).
    *   Generates a detailed dynamic pricing table showing base price, demand level, Z-score, adjustment percentage, and the resulting new price.
*   **Profitability Simulation:**
    *   Compares the estimated total revenue and profit generated using static base prices versus the dynamic pricing strategy based on historical data.
    *   Quantifies the potential uplift from implementing dynamic pricing.
*   **Simulation Environment (Beta Tab):**
    *   Allows users to **upload their own transaction data** (in the expected Excel format).
    *   Lets users set **custom maximum price increase and decrease percentages** via sliders.
    *   Runs the hourly analysis and dynamic pricing calculations on the uploaded data with the specified limits.
    *   Displays the resulting dynamic pricing table and graphs (hourly frequency and dynamic prices) for the simulated scenario.
*   **Interactive Web Application:** Built with Streamlit, providing:
    *   Tabs for organized navigation through different analyses.
    *   Sidebar controls for selecting the number of top items and specific items for detailed views.
    *   Interactive sliders for simulation parameters and MBA thresholds.
    *   Data tables and visualizations (line charts, heatmaps, scatter plots).

## Technologies Used

*   **Language:** Python 3
*   **Data Handling:** Pandas, NumPy
*   **Machine Learning/Data Mining:** mlxtend (for Association Rules), scikit-learn (for data scaling)
*   **Visualization:** Matplotlib, Seaborn
*   **Web Framework:** Streamlit
*   **File Handling:** openpyxl (for reading `.xlsx`)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/belepod/Dynamic-Pricing-WebApp.git
    cd Dynamic-Pricing-WebApp
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` doesn't exist, create one or install manually: `pip install pandas numpy matplotlib seaborn streamlit openpyxl scikit-learn mlxtend`)*

4.  **Prepare Data:**
    *   Place your transaction data Excel file in the main project directory.
    *   Ensure the file is named `dma_data.xlsx` (or update `FILE_PATH` in `app.py`).
    *   Make sure the relevant data is on a sheet named `Sheet1` (or update `SHEET_NAME` in `appdeploy.py`).
    *   The Excel file must contain columns: `Purchased_Items`, `Quantities`, `Prices_per_Unit`, `Time_of_Purchase`. Other columns like `Transaction_ID` are used for logging if present.
    *   `Time_of_Purchase` should be in a format parseable by pandas (e.g., `HH:MM` or `HH:MM:SS`).

5.  **Run the Streamlit App:**
    ```bash
    streamlit run appdeploy.py
    ```
    The application should open automatically in your web browser.

## Code Structure

*   `appdeploy.py`: The main Streamlit application script containing data loading, analysis functions, plotting functions, and the UI layout.
*   `appinit.py`: A test python file for debugging
*   `dma_data.xlsx`: The input Excel data file (ensure this exists and is correctly named/placed).
*   `requirements.txt` (optional but recommended): Lists the Python dependencies.

## Key Assumptions & Customization

*   **Cost Factor:** Profit calculations currently assume the cost of goods is 60% of the selling price (`ASSUMED_COST_FACTOR = 0.60`). Modify this constant in `app.py` if you have actual cost data or want to use a different assumption.
*   **Operating Hours:** The default shop hours are set from 8:00 AM (inclusive) to 9:00 PM (exclusive) via `OPEN_HOUR_START = 8` and `OPEN_HOUR_END = 21`. Change these constants to match the actual operating hours.
*   **Dynamic Pricing Limits:** Default minimum and maximum price change percentages (`MIN_PRICE_CHANGE`, `MAX_PRICE_CHANGE`) are used for the main analysis. The simulation tab allows users to override these.
*   **Demand Calculation:** The dynamic pricing logic uses the Z-score of hourly quantity compared to the *average hourly quantity during open hours* as the primary demand indicator.
*   **MBA Parameters:** Default minimum support and confidence values are set for the Market Basket Analysis but can be adjusted in the app.

## Future Work

*   Incorporate actual cost data for more accurate profit analysis and simulation.
*   Implement more sophisticated demand forecasting models (ARIMA, Prophet) to predict future demand and inform pricing.
*   Consider price elasticity of demand – how much does demand change when the price changes?
*   Add inventory level constraints to the pricing logic (e.g., increase prices for low-stock items).
*   Factor in competitor pricing (if data were available).
*   Use optimization algorithms (e.g., reinforcement learning) to find optimal pricing strategies instead of rule-based ones.
*   Enhance the dashboard with more visualizations (e.g., network graphs for MBA) and potentially user authentication.

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests. Please open an issue first to discuss significant changes.

## License

MIT Licence

---
