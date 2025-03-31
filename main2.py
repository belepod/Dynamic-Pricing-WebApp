import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import time
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide", page_title="Dynamic Pricing System")
st.title("🛒 Dynamic Pricing and Sales Analysis System")

FILE_PATH = "dma_data.xlsx"
SHEET_NAME = "Sheet1"
ASSUMED_COST_FACTOR = 0.60
MIN_PRICE_CHANGE = 0.05
MAX_PRICE_CHANGE = 0.25
OPEN_HOUR_START = 8
OPEN_HOUR_END = 21

@st.cache_data
def load_and_process_data(file_path, sheet_name, cost_factor):
    try:
        xls = pd.ExcelFile(file_path)
        df_raw = pd.read_excel(xls, sheet_name=sheet_name, dtype={"Time_of_Purchase": str})
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Make sure '{file_path}' is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

    required_cols = ["Purchased_Items", "Quantities", "Prices_per_Unit", "Time_of_Purchase"]
    if not all(col in df_raw.columns for col in required_cols):
        st.error(f"Missing one or more required columns in '{sheet_name}'. Expected: {', '.join(required_cols)}")
        return None

    structured_data = []
    for index, row in df_raw.iterrows():
        items_str = str(row.get("Purchased_Items", ""))
        quantities_str = str(row.get("Quantities", ""))
        prices_str = str(row.get("Prices_per_Unit", ""))
        time_str = str(row.get("Time_of_Purchase", ""))
        transaction_id = row.get("Transaction_ID", f"Row_{index+2}")

        items = [i.strip() for i in items_str.split(',') if i.strip()]
        quantities = [q.strip() for q in quantities_str.split(',') if q.strip()]
        prices = [p.strip() for p in prices_str.split(',') if p.strip()]

        len_items = len(items)
        len_qty = len(quantities)
        len_price = len(prices)

        if not (len_items == len_qty == len_price):
            st.warning(f"Data mismatch in {transaction_id}: {len_items} items, {len_qty} quantities, {len_price} prices. Skipping this transaction.")
            continue

        try:
            purchase_time_obj = pd.to_datetime(time_str, errors='coerce').time()
            if pd.isna(purchase_time_obj):
                 try:
                     purchase_time_obj = pd.to_datetime(time_str, format='%H:%M', errors='coerce').time()
                 except ValueError:
                      try:
                          purchase_time_obj = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce').time()
                      except ValueError:
                          purchase_time_obj = pd.NaT

            if pd.isna(purchase_time_obj):
                st.warning(f"Invalid time format '{time_str}' in {transaction_id}. Skipping transaction.")
                continue
            hour = purchase_time_obj.hour
        except Exception as e:
            st.warning(f"Error processing time '{time_str}' in {transaction_id}: {e}. Skipping transaction.")
            continue

        for i in range(len_items):
            try:
                item = items[i]
                quantity = float(quantities[i])
                price = float(prices[i])

                if quantity <= 0 or price < 0:
                     st.warning(f"Invalid quantity ({quantity}) or price ({price}) for item '{item}' in {transaction_id}. Skipping item.")
                     continue

                cost = price * cost_factor
                revenue = quantity * price
                profit = quantity * (price - cost)

                structured_data.append({
                    'Item': item,
                    'Quantity': quantity,
                    'Price': price,
                    'Cost': cost,
                    'Revenue': revenue,
                    'Profit': profit,
                    'Time_of_Purchase': purchase_time_obj,
                    'Hour': hour
                })
            except ValueError:
                st.warning(f"Non-numeric quantity '{quantities[i]}' or price '{prices[i]}' for item '{item}' in {transaction_id}. Skipping item.")
                continue
            except Exception as e:
                 st.error(f"An unexpected error occurred processing item '{item}' in {transaction_id}: {e}")
                 continue

    if not structured_data:
        st.error("No valid data could be processed after cleaning. Please check the Excel file content.")
        return None

    df = pd.DataFrame(structured_data)
    return df

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

@st.cache_data
def prepare_mba_data(file_path, sheet_name):
    """Prepares data specifically for Market Basket Analysis."""
    try:
        xls = pd.ExcelFile(file_path)
        df_raw = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Error reading Excel file for MBA: {e}")
        return None

    if 'Purchased_Items' not in df_raw.columns:
        st.error("Column 'Purchased_Items' not found for MBA.")
        return None

    transactions = []
    for _, row in df_raw.iterrows():
        items_str = str(row.get("Purchased_Items", ""))
        items = [item.strip() for item in items_str.split(',') if item.strip()]
        if items:
             transactions.append(items)

    if not transactions:
         st.warning("No transactions found for Market Basket Analysis.")
         return None

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    return df_encoded

@st.cache_data
def run_market_basket_analysis(df_encoded, min_support=0.01, min_confidence=0.1):
    """Runs Apriori and generates association rules."""
    if df_encoded is None or df_encoded.empty:
        return pd.DataFrame(), pd.DataFrame()

    try:
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            st.warning(f"No frequent itemsets found with minimum support {min_support}. Try lowering it.")
            return frequent_itemsets, pd.DataFrame()

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            st.warning(f"No association rules found with minimum confidence {min_confidence}. Try lowering it.")
            return frequent_itemsets, rules

        rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])

        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

        return frequent_itemsets, rules

    except Exception as e:
        st.error(f"Error during Market Basket Analysis: {e}")
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data
def get_hourly_financials(df):
    """Aggregates revenue and profit by hour and item."""
    if df is None or df.empty or 'Hour' not in df.columns or 'Item' not in df.columns \
       or 'Revenue' not in df.columns or 'Profit' not in df.columns:
        return pd.DataFrame(columns=['Hour', 'Item', 'TotalRevenue', 'TotalProfit'])

    all_hours_items = pd.MultiIndex.from_product(
        [df['Item'].unique(), range(24)], names=['Item', 'Hour']
    )

    hourly_financials = df.groupby(['Item', 'Hour'])[['Revenue', 'Profit']].sum().reindex(all_hours_items, fill_value=0).reset_index()
    hourly_financials.rename(columns={'Revenue': 'TotalRevenue', 'Profit': 'TotalProfit'}, inplace=True)

    return hourly_financials

def plot_hourly_financial_trend(df_financial, items_to_plot, value_col, title, ylabel, open_start, open_end):
    """Plots hourly revenue or profit trends for selected items ONLY during open hours."""
    if df_financial.empty or not items_to_plot or value_col not in df_financial.columns:
        st.warning(f"No data available for plotting {ylabel} trends.")
        return

    filtered_data = df_financial[df_financial['Item'].isin(items_to_plot)]
    if filtered_data.empty:
         st.warning(f"No {ylabel} data found for the selected items: {', '.join(items_to_plot)}")
         return

    plot_data = filtered_data[(filtered_data['Hour'] >= open_start) & (filtered_data['Hour'] < open_end)]

    fig, ax = plt.subplots(figsize=(12, 6))

    if not plot_data.empty and plot_data[value_col].sum() > 0:
        sns.lineplot(data=plot_data, x="Hour", y=value_col, hue="Item", marker="o", ax=ax)
    else:
        st.warning(f"No {ylabel} recorded during open hours ({open_start:02d}:00 - {open_end:02d}:00) for selected items.")
        ax.set_ylabel(ylabel)

    hours_range = range(24)
    for h in hours_range:
        if not (open_start <= h < open_end):
            ax.axvspan(h - 0.5, h + 0.5, color='gray', alpha=0.15, zorder=0)

    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)

    ax.set_xticklabels([f"{h:02d}:00" for h in hours_range], rotation=45, ha="right")
    ax.set_xlabel("Hour of Purchase (Full Day Context)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not plot_data.empty and plot_data[value_col].sum() > 0:
        ax.legend(title="Item", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, axis='y', linestyle='--')
    st.pyplot(fig)


def plot_total_financials(df_financial, value_col, title, xlabel):
    """Plots total revenue or profit by item using a bar chart."""
    if df_financial.empty or value_col not in df_financial.columns:
        st.warning(f"No data available for plotting total {xlabel}.")
        return

    total_financial_by_item = df_financial.groupby('Item')[value_col].sum().reset_index()
    total_financial_by_item = total_financial_by_item[total_financial_by_item[value_col] > 0]
    total_financial_by_item = total_financial_by_item.sort_values(value_col, ascending=False).head(20)

    if total_financial_by_item.empty:
         st.warning(f"No positive {xlabel} found to plot.")
         return

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=total_financial_by_item, x=value_col, y='Item', palette='viridis', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Item')
    st.pyplot(fig)

@st.cache_data
def get_hourly_sales(df):
    """Aggregates sales quantity by hour and item."""
    if df is None or 'Hour' not in df.columns or 'Item' not in df.columns or 'Quantity' not in df.columns:
        return pd.DataFrame(columns=['Hour', 'Item', 'Quantity'])

    all_hours_items = pd.MultiIndex.from_product(
        [df['Item'].unique(), range(24)], names=['Item', 'Hour']
    )
    hourly_analysis = df.groupby(['Item', 'Hour'])['Quantity'].sum().reindex(all_hours_items, fill_value=0).reset_index()
    return hourly_analysis

@st.cache_data
def get_item_stats(df_hourly, open_start, open_end):
    """
    Calculates total quantity and average/std dev of hourly sales
    during OPEN hours for items.
    """
    if df_hourly.empty or 'Item' not in df_hourly.columns or 'Quantity' not in df_hourly.columns:
         return pd.DataFrame(columns=['Item', 'TotalQuantity', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen'])

    total_sales = df_hourly.groupby('Item')['Quantity'].sum().reset_index().rename(columns={'Quantity':'TotalQuantity'})

    open_hours_sales = df_hourly[(df_hourly['Hour'] >= open_start) & (df_hourly['Hour'] < open_end)]

    if open_hours_sales.empty:
        st.warning("No sales data found during open hours to calculate item statistics (Avg/StdDev).")
        open_stats = pd.DataFrame({
            'Item': df_hourly['Item'].unique(),
            'AvgHourlyQtyOpen': 0.0,
            'StdDevHourlyQtyOpen': 0.0
        })
    else:
        open_stats = open_hours_sales.groupby('Item')['Quantity'].agg(
            AvgHourlyQtyOpen='mean',
            StdDevHourlyQtyOpen='std'
        ).reset_index()
        open_stats['StdDevHourlyQtyOpen'] = open_stats['StdDevHourlyQtyOpen'].fillna(0)

    item_performance = pd.merge(total_sales, open_stats, on='Item', how='left')
    item_performance['AvgHourlyQtyOpen'] = item_performance['AvgHourlyQtyOpen'].fillna(0.0)
    item_performance['StdDevHourlyQtyOpen'] = item_performance['StdDevHourlyQtyOpen'].fillna(0.0)

    item_performance['SalesStdDev'] = item_performance['StdDevHourlyQtyOpen']


    return item_performance[['Item', 'TotalQuantity', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen', 'SalesStdDev']]

@st.cache_data
def categorize_items(item_stats):
    """Categorizes items based on sales volume and consistency using quantiles."""
    if item_stats.empty:
        return item_stats

    qty_median = item_stats['TotalQuantity'].quantile(0.5)
    std_median = item_stats['SalesStdDev'].quantile(0.5)

    conditions = [
        (item_stats['TotalQuantity'] >= qty_median) & (item_stats['SalesStdDev'] >= std_median),
        (item_stats['TotalQuantity'] >= qty_median) & (item_stats['SalesStdDev'] < std_median),
        (item_stats['TotalQuantity'] < qty_median) & (item_stats['SalesStdDev'] >= std_median),
        (item_stats['TotalQuantity'] < qty_median) & (item_stats['SalesStdDev'] < std_median)
    ]
    categories = ['High Vol, Erratic', 'High Vol, Stable (Star)', 'Low Vol, Erratic (QM)', 'Low Vol, Stable (Dog)']

    item_stats['Category'] = np.select(conditions, categories, default='Undefined')
    return item_stats


@st.cache_data
def calculate_dynamic_pricing(df, df_hourly, max_increase_pct, max_decrease_pct, open_start, open_end):
    """
    Calculates dynamic pricing adjustments based on hourly demand relative to
    the item's average hourly demand during open hours, using z-scores.
    """
    if df is None or df.empty or df_hourly.empty:
        return pd.DataFrame(), pd.DataFrame()

    max_increase_factor = max_increase_pct / 100.0
    max_decrease_factor = max_decrease_pct / 100.0

    base_prices = df.groupby('Item')['Price'].median().reset_index()
    base_prices.rename(columns={'Price': 'BasePrice'}, inplace=True)

    open_hours_sales = df_hourly[(df_hourly['Hour'] >= open_start) & (df_hourly['Hour'] < open_end)]

    if open_hours_sales.empty:
        st.warning("No sales data found during specified open hours. Cannot calculate demand stats or dynamic prices.")
        item_hourly_stats = pd.DataFrame(columns=['Item', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen'])
        dynamic_pricing_final = pd.DataFrame(columns=['Item', 'Hour', 'Status', 'BasePrice',
                                                     'DemandLevel (Qty)', 'DemandZScore', 'Adjustment', 'NewPrice'])
        return dynamic_pricing_final, item_hourly_stats

    item_hourly_stats = open_hours_sales.groupby('Item')['Quantity'].agg(
        AvgHourlyQtyOpen='mean',
        StdDevHourlyQtyOpen='std'
    ).reset_index()
    item_hourly_stats['StdDevHourlyQtyOpen'] = item_hourly_stats['StdDevHourlyQtyOpen'].fillna(0)


    pricing_data = []
    all_items = df['Item'].unique()
    hours_range = range(24)

    hourly_sales_with_stats = pd.merge(df_hourly, item_hourly_stats, on='Item', how='left')
    hourly_sales_with_stats['AvgHourlyQtyOpen'] = hourly_sales_with_stats['AvgHourlyQtyOpen'].fillna(0)
    hourly_sales_with_stats['StdDevHourlyQtyOpen'] = hourly_sales_with_stats['StdDevHourlyQtyOpen'].fillna(0)

    stats_lookup = hourly_sales_with_stats.set_index(['Item', 'Hour']).to_dict('index')

    for item in all_items:
        if item not in base_prices['Item'].values:
            continue
        base_price = base_prices.loc[base_prices['Item'] == item, 'BasePrice'].iloc[0]

        for hour in hours_range:
            is_open = open_start <= hour < open_end
            status = "Open" if is_open else "Closed"
            adjustment_pct = 0.0
            z_score = np.nan
            current_qty = 0

            lookup_key = (item, hour)
            hour_data = stats_lookup.get(lookup_key)

            if hour_data:
                 current_qty = hour_data.get('Quantity', 0)
                 avg_qty = hour_data.get('AvgHourlyQtyOpen', 0)
                 std_dev = hour_data.get('StdDevHourlyQtyOpen', 0)

                 if is_open:
                     if std_dev > 0:
                         z_score = (current_qty - avg_qty) / std_dev
                     elif avg_qty > 0:
                         z_score = 0

                     z_min, z_max = -1.5, 1.5
                     adj_min, adj_max = -max_decrease_factor, max_increase_factor

                     if pd.notna(z_score):
                           adjustment_pct = np.interp(z_score, [z_min, z_max], [adj_min, adj_max])


            adjustment_pct = np.clip(adjustment_pct, -max_decrease_factor, max_increase_factor)
            new_price = round(base_price * (1 + adjustment_pct), 2)
            adj_str = f"{int(round(adjustment_pct * 100))}%" if is_open else "0% (Closed)"

            pricing_data.append({
                'Item': item,
                'Hour': f"{hour:02d}:00",
                'Status': status,
                'BasePrice': round(base_price, 2),
                'DemandLevel (Qty)': current_qty,
                'DemandZScore': round(z_score, 2) if pd.notna(z_score) else np.nan,
                'Adjustment': adj_str,
                'NewPrice': new_price
            })

    dynamic_pricing_final = pd.DataFrame(pricing_data)

    dynamic_pricing_final = pd.merge(dynamic_pricing_final, item_hourly_stats[['Item', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen']], on='Item', how='left')

    return dynamic_pricing_final, item_hourly_stats

@st.cache_data
def simulate_profitability(df, pricing_table):
    """Simulates revenue and profit with static vs dynamic pricing."""
    if df is None or df.empty or pricing_table.empty:
        return 0, 0, 0, 0

    pricing_map = pricing_table.set_index(['Item', 'Hour'])['NewPrice'].to_dict()
    base_price_map = pricing_table.set_index('Item')['BasePrice'].to_dict()

    total_static_revenue = 0
    total_static_profit = 0
    total_dynamic_revenue = 0
    total_dynamic_profit = 0

    for _, row in df.iterrows():
        item = row['Item']
        hour_int = row['Hour']
        hour_str = f"{hour_int:02d}:00"
        quantity = row['Quantity']
        cost = row['Cost']

        base_price = base_price_map.get(item, 0)
        static_revenue_txn = quantity * base_price
        static_profit_txn = quantity * (base_price - cost)
        total_static_revenue += static_revenue_txn
        total_static_profit += static_profit_txn

        dynamic_price_key = (item, hour_str)
        dynamic_price = pricing_map.get(dynamic_price_key, base_price)
        dynamic_revenue_txn = quantity * dynamic_price
        dynamic_profit_txn = quantity * (dynamic_price - cost)
        total_dynamic_revenue += dynamic_revenue_txn
        total_dynamic_profit += dynamic_profit_txn

    return total_static_revenue, total_static_profit, total_dynamic_revenue, total_dynamic_profit


def plot_hourly_trend(df_hourly, items_to_plot, title, open_start, open_end, item_stats_sim=None):
    """Plots hourly sales quantity trends for selected items ONLY during open hours, with average line."""
    if df_hourly.empty or not items_to_plot:
        st.warning("No data available for plotting hourly trends.")
        return

    filtered_data = df_hourly[df_hourly['Item'].isin(items_to_plot)]
    if filtered_data.empty:
         st.warning(f"No hourly data found for the selected items: {', '.join(items_to_plot)}")
         return

    plot_data = filtered_data[(filtered_data['Hour'] >= open_start) & (filtered_data['Hour'] < open_end)]

    fig, ax = plt.subplots(figsize=(12, 6))

    if not plot_data.empty:
        sns.lineplot(data=plot_data, x="Hour", y="Quantity", hue="Item", marker="o", ax=ax)

        if item_stats_sim is not None and not item_stats_sim.empty:
            items_in_plot = plot_data['Item'].unique()
            stats_to_plot = item_stats_sim[item_stats_sim['Item'].isin(items_in_plot)]
            palette = sns.color_palette(n_colors=len(items_in_plot))
            item_color_map = dict(zip(items_in_plot, palette))

            for _, item_row in stats_to_plot.iterrows():
                item_name = item_row['Item']
                avg_qty = item_row['AvgHourlyQtyOpen']
                if item_name in item_color_map:
                    ax.axhline(y=avg_qty, color=item_color_map[item_name], linestyle=':',
                               label=f'{item_name} Avg (Open Hrs)')
    else:
        st.warning(f"No sales recorded during open hours ({open_start:02d}:00 - {open_end:02d}:00) for selected items.")
        ax.set_ylabel("Total Quantity Sold")

    hours_range = range(24)
    for h in hours_range:
        if not (open_start <= h < open_end):
            ax.axvspan(h - 0.5, h + 0.5, color='gray', alpha=0.15, zorder=0)

    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)

    ax.set_xticklabels([f"{h:02d}:00" for h in hours_range], rotation=45, ha="right")
    ax.set_xlabel("Hour of Purchase (Full Day Context)")
    ax.set_ylabel("Total Quantity Sold")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
            unique_handles.append(handle)

    ax.legend(unique_handles, unique_labels.keys(), title="Item / Avg", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, axis='y', linestyle='--')
    st.pyplot(fig)

def plot_sales_heatmap(df_hourly, top_n=15, open_start=0, open_end=24):
    """Plots a heatmap of hourly sales for top N items, with lines for open hours."""
    if df_hourly.empty:
         st.warning("No data available for plotting heatmap.")
         return

    top_items = df_hourly.groupby('Item')['Quantity'].sum().nlargest(top_n).index
    heatmap_data = df_hourly[df_hourly['Item'].isin(top_items)]

    if heatmap_data.empty:
        st.warning("No data found for the top items to generate heatmap.")
        return

    heatmap_pivot = heatmap_data.pivot_table(index='Item', columns='Hour', values='Quantity', fill_value=0)
    heatmap_pivot = heatmap_pivot.reindex(columns=range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(15, max(6, len(top_items)*0.4)))
    sns.heatmap(heatmap_pivot, cmap="viridis", linewidths=.5, ax=ax, annot=False, cbar_kws={'label': 'Quantity Sold'})

    ax.axvline(x=open_start, color='white', linestyle='--', linewidth=2)
    ax.axvline(x=open_end, color='white', linestyle='--', linewidth=2)

    ax.set_title(f'Hourly Sales Heatmap (Top {top_n} Items)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Item')
    ax.set_xticks(np.arange(24) + 0.5)
    ax.set_xticklabels(range(24))
    plt.yticks(rotation=0)
    st.pyplot(fig)

def plot_item_categorization(item_stats):
    """Plots the item categorization scatter plot."""
    if item_stats.empty:
         st.warning("No data available for plotting item categories.")
         return

    scaler = MinMaxScaler()
    stats_scaled = item_stats.copy()
    cols_to_scale = ['TotalQuantity', 'SalesStdDev']
    if all(col in stats_scaled.columns for col in cols_to_scale):
         stats_scaled[cols_to_scale] = scaler.fit_transform(stats_scaled[cols_to_scale])
         x_col, y_col = 'TotalQuantity', 'SalesStdDev'
         x_label, y_label = "Normalized Total Quantity", "Normalized Sales Std Dev"
    else:
         st.warning("Required columns for scaling not found.")
         return


    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=stats_scaled, x=x_col, y=y_col, hue='Category', style='Category', s=100, ax=ax)
    ax.set_title('Item Performance Categorization')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--')


    st.pyplot(fig)

def plot_pricing_strategy(pricing_df, item, open_start, open_end):
    """Plots Z-score demand vs adjusted price for a specific item ONLY during open hours."""
    item_pricing = pricing_df[pricing_df['Item'] == item].sort_values('Hour')
    if item_pricing.empty:
        st.warning(f"No pricing data found for item: {item}")
        return

    hours_str = item_pricing['Hour'].tolist()
    hours_int = [int(h.split(':')[0]) for h in hours_str]
    prices = item_pricing['NewPrice'].tolist()
    demand_zscores = item_pricing['DemandZScore'].tolist()
    base_price = item_pricing['BasePrice'].iloc[0]

    open_hours_indices = [i for i, h in enumerate(hours_int) if open_start <= h < open_end]
    open_hours_int = [hours_int[i] for i in open_hours_indices]
    open_prices = [prices[i] for i in open_hours_indices]
    open_zscores = [demand_zscores[i] for i in open_hours_indices if pd.notna(demand_zscores[i])]
    open_hours_zscore_plot = [h for h, z in zip(open_hours_int, [demand_zscores[i] for i in open_hours_indices]) if pd.notna(z)]


    fig, ax1 = plt.subplots(figsize=(14, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Hour of Day (Full Day Context)')
    ax1.set_ylabel('Demand Z-Score', color=color)
    if open_hours_zscore_plot:
        ax1.plot(open_hours_zscore_plot, open_zscores, color=color, marker='o', linestyle='--', label='Demand Z-Score')
        ax1.axhline(y=0, color=color, linestyle=':', alpha=0.5, label='Avg Demand (Z=0)')
    ax1.tick_params(axis='y', labelcolor=color)
    if open_zscores:
         min_z, max_z = min(open_zscores), max(open_zscores)
         padding = max(1, (max_z - min_z) * 0.1)
         ax1.set_ylim(min_z - padding, max_z + padding)
    else:
         ax1.set_ylim(-2, 2)


    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Price ($)', color=color)
    if open_hours_int:
        ax2.plot(open_hours_int, open_prices, color=color, marker='x', linestyle='-', label='Dynamic Price')
    ax2.axhline(y=base_price, color='grey', linestyle=':', label=f'Base Price (${base_price:.2f})')
    ax2.tick_params(axis='y', labelcolor=color)

    hours_range = range(24)
    for h in hours_range:
        if not (open_start <= h < open_end):
            ax1.axvspan(h - 0.5, h + 0.5, color='gray', alpha=0.15, zorder=0)

    ax1.set_xticks(range(24))
    ax1.set_xticklabels([f"{h:02d}:00" for h in hours_range], rotation=45, ha="right")
    ax1.set_xlim(-0.5, 23.5)
    ax1.grid(True, axis='y', linestyle=':')
    ax2.grid(False)

    fig.suptitle(f'Dynamic Pricing Strategy for {item} (Shop Hours: {open_start:02d}:00-{open_end:02d}:00)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.93, 0.95))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

def plot_dynamic_prices_sim(pricing_df, items_to_plot, title, open_start, open_end):
    """Plots the calculated dynamic prices over 24 hours for selected items ONLY during open hours."""
    if pricing_df.empty or not items_to_plot:
        st.warning("No pricing data available for plotting.")
        return

    filtered_pricing = pricing_df[pricing_df['Item'].isin(items_to_plot)].copy()
    if filtered_pricing.empty:
        st.warning(f"No pricing data found for the selected items: {', '.join(items_to_plot)}")
        return

    filtered_pricing['HourInt'] = filtered_pricing['Hour'].apply(lambda x: int(x.split(':')[0]))

    plot_data = filtered_pricing[(filtered_pricing['HourInt'] >= open_start) & (filtered_pricing['HourInt'] < open_end)]

    fig, ax = plt.subplots(figsize=(12, 6))

    if not plot_data.empty:
         sns.lineplot(data=plot_data, x="HourInt", y="NewPrice", hue="Item", marker="o", ax=ax)
         base_prices_plot = plot_data.drop_duplicates(subset=['Item'])
         palette = sns.color_palette(n_colors=len(items_to_plot))
         item_color_map = dict(zip(base_prices_plot['Item'], palette))
         for _, item_row in base_prices_plot.iterrows():
             item_name = item_row['Item']
             base_price_val = item_row['BasePrice']
             if item_name in item_color_map:
                  ax.axhline(y=base_price_val, color=item_color_map[item_name], linestyle=':', label=f'{item_name} Base')
    else:
         st.warning(f"No dynamic pricing calculated during open hours ({open_start:02d}:00 - {open_end:02d}:00) for selected items.")
         ax.set_ylabel("Calculated Dynamic Price ($)")


    hours_range = range(24)
    for h in hours_range:
        if not (open_start <= h < open_end):
            ax.axvspan(h - 0.5, h + 0.5, color='gray', alpha=0.15, zorder=0)

    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}:00" for h in hours_range], rotation=45, ha="right")
    ax.set_xlim(-0.5, 23.5)
    if not plot_data.empty:
         min_price = plot_data['NewPrice'].min()
         max_price = plot_data['NewPrice'].max()
         ax.set_ylim(bottom=min(0, min_price * 0.9), top=max_price * 1.1)
    else:
         ax.set_ylim(bottom=0)


    ax.set_xlabel("Hour of Day (Full Day Context)")
    ax.set_ylabel("Calculated Dynamic Price ($)")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels.keys(), title="Item / Base", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.grid(True, axis='y', linestyle='--')
    st.pyplot(fig)

df_processed = load_and_process_data(FILE_PATH, SHEET_NAME, ASSUMED_COST_FACTOR)

if df_processed is not None:
    hourly_sales_base = get_hourly_sales(df_processed)
    item_stats_base = get_item_stats(hourly_sales_base, OPEN_HOUR_START, OPEN_HOUR_END)
    item_categories_base = categorize_items(item_stats_base)
    dynamic_pricing_base, _ = calculate_dynamic_pricing(
        df_processed, hourly_sales_base, MIN_PRICE_CHANGE, MAX_PRICE_CHANGE, OPEN_HOUR_START, OPEN_HOUR_END
    )
    sim_results_base = simulate_profitability(df_processed, dynamic_pricing_base)

    hourly_financials_base = get_hourly_financials(df_processed)

    mba_encoded_data = prepare_mba_data(FILE_PATH, SHEET_NAME)
    frequent_itemsets_base, rules_base = run_market_basket_analysis(mba_encoded_data)


    st.sidebar.header("Filters & Options (Main Analysis)")
    all_items_sorted_qty = item_stats_base.sort_values("TotalQuantity", ascending=False)['Item'].tolist()
    max_slider_value = min(20, len(all_items_sorted_qty))
    if max_slider_value < 1:
        top_n_items = 0
        selected_top_items = []
        item_for_detail = None
        st.sidebar.warning("No item data for main analysis.")
    else:
        top_n_items = st.sidebar.slider(
            "Select No. of Top Items (Sales Qty):",
            min_value=1,
            max_value=max_slider_value,
            value=min(5, max_slider_value),
            key="slider_main_qty"
        )
        selected_top_items = all_items_sorted_qty[:top_n_items]

        item_for_detail = st.sidebar.selectbox(
            "Select Item for Detailed View:",
            all_items_sorted_qty,
            key="select_main_detail"
        )

    st.sidebar.info(f"Shop Operating Hours: {OPEN_HOUR_START:02d}:00 - {OPEN_HOUR_END:02d}:00")

    tab_list = [
        "📊 Sales Overview",
        "💰 Revenue & Profit",
        "🧺 Market Basket",
        "🏷️ Item Deep Dive",
        "📈 Dynamic Pricing",
        "⭐ Item Categories",
        "💰 Profit Simulation",
        "🚀 Simulation (Beta)"
    ]
    tabs = st.tabs(tab_list)

    tab_overview = tabs[0]
    tab_revenue_profit = tabs[1]
    tab_market_basket = tabs[2]
    tab_deep_dive = tabs[3]
    tab_dynamic_pricing = tabs[4]
    tab_categories = tabs[5]
    tab_profit_sim = tabs[6]
    tab_simulation = tabs[7]


    with tab_overview:
        st.header("Sales Overview (Based on Original Data)")

        st.subheader(f"Hourly Sales Trend (Top {top_n_items} by Quantity)")
        if selected_top_items:
            plot_hourly_trend(hourly_sales_base, selected_top_items, f"Hourly Sales Trend for Top {top_n_items} Items (Qty)", OPEN_HOUR_START, OPEN_HOUR_END, item_stats_base)
        else:
             st.write("No items selected or available.")

        st.subheader("Overall Hourly Sales (All Items)")
        overall_hourly_qty = df_processed.groupby('Hour')['Quantity'].sum().reset_index()
        if not overall_hourly_qty.empty:
             fig_overall, ax_overall = plt.subplots(figsize=(12, 5))
             sns.lineplot(data=overall_hourly_qty, x="Hour", y="Quantity", marker="o", ax=ax_overall)
             for h in range(24):
                 if not (OPEN_HOUR_START <= h < OPEN_HOUR_END):
                     ax_overall.axvspan(h - 0.5, h + 0.5, color='gray', alpha=0.15, zorder=0)
             ax_overall.set_xticks(range(24))
             ax_overall.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")
             ax_overall.set_xlabel("Hour of Purchase")
             ax_overall.set_ylabel("Total Quantity Sold (All Items)")
             ax_overall.grid(True, axis='y', linestyle='--')
             st.pyplot(fig_overall)
        else:
             st.write("No overall hourly quantity data available.")

        st.subheader(f"Hourly Sales Heatmap (Top {min(15, len(all_items_sorted_qty))} by Quantity)")
        if not hourly_sales_base.empty:
            plot_sales_heatmap(hourly_sales_base, top_n=min(15, len(all_items_sorted_qty)), open_start=OPEN_HOUR_START, open_end=OPEN_HOUR_END)
        else:
             st.write("No hourly data available for heatmap.")

    with tab_revenue_profit:
        st.header("Revenue & Profit Analysis (Original Data)")

        if not hourly_financials_base.empty:
            top_revenue_items = hourly_financials_base.groupby('Item')['TotalRevenue'].sum().nlargest(top_n_items).index.tolist()
            top_profit_items = hourly_financials_base.groupby('Item')['TotalProfit'].sum().nlargest(top_n_items).index.tolist()

            st.subheader(f"Hourly Revenue Trend (Top {top_n_items} by Revenue)")
            if top_revenue_items:
                plot_hourly_financial_trend(hourly_financials_base, top_revenue_items, 'TotalRevenue', f"Hourly Revenue Trend (Top {top_n_items} by Revenue)", "Total Revenue ($)", OPEN_HOUR_START, OPEN_HOUR_END)
            else:
                st.write("No revenue data for top items.")


            st.subheader(f"Hourly Profit Trend (Top {top_n_items} by Profit)")
            if top_profit_items:
                plot_hourly_financial_trend(hourly_financials_base, top_profit_items, 'TotalProfit', f"Hourly Profit Trend (Top {top_n_items} by Profit)", "Total Profit ($)", OPEN_HOUR_START, OPEN_HOUR_END)
            else:
                 st.write("No profit data for top items.")

            col_rev, col_prof = st.columns(2)
            with col_rev:
                st.subheader("Total Revenue by Item (Top 20)")
                plot_total_financials(hourly_financials_base, 'TotalRevenue', "Total Revenue by Item", "Total Revenue ($)")
            with col_prof:
                st.subheader("Total Profit by Item (Top 20)")
                plot_total_financials(hourly_financials_base, 'TotalProfit', "Total Profit by Item", "Total Profit ($)")
        else:
            st.warning("Could not calculate hourly financial data.")

    with tab_market_basket:
        st.header("Market Basket Analysis")
        st.write("Discovering which items are frequently purchased together using association rules.")

        if mba_encoded_data is not None and not mba_encoded_data.empty:
             min_supp = st.slider("Minimum Support", 0.005, 0.1, 0.01, 0.001, format="%.3f", key="mba_support")
             min_conf = st.slider("Minimum Confidence", 0.05, 0.5, 0.1, 0.01, format="%.2f", key="mba_confidence")

             frequent_itemsets_base, rules_base = run_market_basket_analysis(mba_encoded_data, min_support=min_supp, min_confidence=min_conf)


             st.subheader("Frequent Itemsets")
             if not frequent_itemsets_base.empty:
                 st.dataframe(frequent_itemsets_base.sort_values("support", ascending=False))
             else:
                 st.write("No frequent itemsets found with the current settings.")

             st.subheader("Association Rules")
             if not rules_base.empty:
                 st.dataframe(rules_base[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False))
                 st.markdown("""
                 **Explanation:**
                 - **Antecedents:** Item(s) bought ('If...')
                 - **Consequents:** Item(s) also bought ('...then...')
                 - **Support:** How frequently the itemset appears in all transactions.
                 - **Confidence:** How often the rule is true (if A, then B).
                 - **Lift:** How much more likely B is bought when A is bought, compared to B being bought alone (Lift > 1 suggests association).
                 """)
             else:
                 st.write("No association rules found with the current settings.")
        else:
            st.warning("Could not prepare data for Market Basket Analysis. Check source file.")


    with tab_deep_dive:
        st.header(f"Deep Dive (Original Data): {item_for_detail if item_for_detail else 'No Item Selected'}")
        if item_for_detail:
            st.subheader("Hourly Sales Trend")
            plot_hourly_trend(hourly_sales_base, [item_for_detail], f"Hourly Sales Trend for {item_for_detail}", OPEN_HOUR_START, OPEN_HOUR_END, item_stats_base)

            st.subheader("Dynamic Pricing Strategy")
            if not dynamic_pricing_base.empty:
                plot_pricing_strategy(dynamic_pricing_base, item_for_detail, OPEN_HOUR_START, OPEN_HOUR_END)
            else:
                st.warning("Dynamic pricing table not available for this item.")

            st.subheader("Performance Category")
            category_info = item_categories_base[item_categories_base['Item'] == item_for_detail]
            if not category_info.empty:
                st.metric(label="Category", value=category_info['Category'].iloc[0])
                st.metric(label="Total Quantity Sold", value=f"{category_info['TotalQuantity'].iloc[0]:,.0f}")
                st.metric(label="Sales Std Dev (Open Hrs)", value=f"{category_info['SalesStdDev'].iloc[0]:.2f}")
            else:
                st.warning(f"Category information not found for {item_for_detail}.")
        else:
             st.write("Please select an item from the sidebar for a detailed view.")


    with tab_dynamic_pricing:
        st.header("Dynamic Pricing Recommendations (Original Data)")
        st.write(f"Based on hourly demand relative to average demand *during open hours* ({OPEN_HOUR_START:02d}:00 - {OPEN_HOUR_END:02d}:00). Uses Z-score scaling.")
        st.write(f"Default price adjustments range from {-int(MIN_PRICE_CHANGE*100)}% to +{int(MAX_PRICE_CHANGE*100)}%.")

        if not dynamic_pricing_base.empty:
             show_all_base = st.checkbox("Show base pricing for all items?", value=False, key="cb_base_all")
             display_df_base = dynamic_pricing_base
             if not show_all_base and selected_top_items:
                 display_df_base = dynamic_pricing_base[dynamic_pricing_base['Item'].isin(selected_top_items)]

             if not display_df_base.empty:
                  cols_to_show_base = ['Item', 'Hour', 'Status', 'BasePrice', 'DemandLevel (Qty)', 'DemandZScore', 'Adjustment', 'NewPrice', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen']
                  st.dataframe(display_df_base[cols_to_show_base], height=500)
             else:
                  st.write("No data to display based on selection.")
        else:
             st.warning("Dynamic pricing table (base) could not be generated.")


    with tab_categories:
        st.header("Item Performance Categories (Original Data)")
        st.write("Items categorized by total sales volume and sales consistency (standard deviation across *open* hours).")
        if not item_categories_base.empty:
            plot_item_categorization(item_categories_base)

            st.subheader("Category Details")
            st.dataframe(item_categories_base[['Item', 'TotalQuantity', 'SalesStdDev', 'Category']].sort_values('TotalQuantity', ascending=False), height=400)
        else:
             st.warning("Item category data not available.")


    with tab_profit_sim:
        st.header("Profitability Simulation (Original Data)")
        st.write("Comparing estimated total revenue and profit using static base prices versus the default dynamic pricing strategy.")
        st.warning(f"Note: Profit calculation assumes a cost factor of {ASSUMED_COST_FACTOR:.0%}. Replace with actual costs for real analysis.")

        stat_rev, stat_prof, dyn_rev, dyn_prof = sim_results_base

        if stat_rev > 0 or dyn_rev > 0:
            if stat_rev > 0:
                rev_uplift = ((dyn_rev - stat_rev) / stat_rev) * 100
                rev_delta_text = f"{rev_uplift:.1f}%"
            else:
                rev_uplift = float('inf')
                rev_delta_text = "N/A (From $0)"

            if stat_prof == 0 and dyn_prof == 0:
                prof_uplift = 0
                prof_delta_text = "0.0%"
            elif stat_prof <= 0 and dyn_prof > 0:
                 prof_uplift = float('inf')
                 prof_delta_text = "N/A (Profit Increased)"
            elif stat_prof > 0:
                 prof_uplift = ((dyn_prof - stat_prof) / stat_prof) * 100
                 prof_delta_text = f"{prof_uplift:.1f}%"
            else:
                 prof_uplift = 0
                 prof_delta_text = "N/A (Loss Context)"

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Revenue")
                st.metric(label="Static Revenue", value=f"${stat_rev:,.2f}")
                st.metric(label="Dynamic Revenue", value=f"${dyn_rev:,.2f}", delta=rev_delta_text)

            with col2:
                st.subheader("Profit (Estimated)")
                st.metric(label="Static Profit", value=f"${stat_prof:,.2f}")
                st.metric(label="Dynamic Profit", value=f"${dyn_prof:,.2f}", delta=prof_delta_text)

            st.markdown("---")
            st.write("Simulation uses historical quantities and applies either the base price or the calculated dynamic price for each transaction's hour.")
        else:
            st.warning("Could not perform profitability simulation on original data.")


    with tab_simulation:
        st.header("🚀 Simulation Environment (Beta)")
        st.info("Upload an Excel file with the same format as the original data to run a simulation with custom dynamic pricing limits.")

        uploaded_file = st.file_uploader("Upload Simulation Data (.xlsx)", type="xlsx", key="sim_upload")

        st.markdown("---")
        st.subheader("Set Simulation Pricing Limits:")
        max_decrease_pct_sim = st.slider("Max Price Decrease (%)", min_value=0, max_value=50, value=int(MIN_PRICE_CHANGE*100), step=1, key="slider_sim_decrease")
        max_increase_pct_sim = st.slider("Max Price Increase (%)", min_value=0, max_value=50, value=int(MAX_PRICE_CHANGE*100), step=1, key="slider_sim_increase")

        if uploaded_file is not None:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")

            df_sim = load_and_process_data(uploaded_file, SHEET_NAME, ASSUMED_COST_FACTOR)

            if df_sim is not None and not df_sim.empty:
                st.markdown("---")
                st.subheader("Simulation Results")

                hourly_sales_sim = get_hourly_sales(df_sim)
                dynamic_pricing_sim, item_stats_sim = calculate_dynamic_pricing(
                    df_sim, hourly_sales_sim, max_increase_pct_sim, max_decrease_pct_sim, OPEN_HOUR_START, OPEN_HOUR_END
                )

                if not dynamic_pricing_sim.empty:
                    if not item_stats_sim.empty:
                         total_qty_sim = hourly_sales_sim.groupby('Item')['Quantity'].sum().reset_index().rename(columns={'Quantity':'TotalQuantity'})
                         item_stats_sim_merged = pd.merge(item_stats_sim, total_qty_sim, on='Item', how='left').fillna(0)
                         top_items_sim_list = item_stats_sim_merged.sort_values("TotalQuantity", ascending=False)['Item'].head(5).tolist()
                    else:
                         top_items_sim_list = []
                         st.warning("Could not determine demand stats or top items from simulation data.")


                    st.write(f"Dynamic Pricing Table (Using {max_decrease_pct_sim}% Decrease / {max_increase_pct_sim}% Increase limits)")
                    cols_to_show_sim = ['Item', 'Hour', 'Status', 'BasePrice', 'DemandLevel (Qty)', 'DemandZScore', 'Adjustment', 'NewPrice', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen']
                    st.dataframe(dynamic_pricing_sim[cols_to_show_sim], height=400)

                    if top_items_sim_list:
                         st.write("Hourly Sales Frequency Graph (Top 5 from Uploaded Data)")
                         plot_hourly_trend(hourly_sales_sim, top_items_sim_list, "Hourly Sales Trend (Simulation Data - Top 5)", OPEN_HOUR_START, OPEN_HOUR_END, item_stats_sim)

                         st.write("Calculated Dynamic Prices Graph (Top 5 from Uploaded Data)")
                         plot_dynamic_prices_sim(dynamic_pricing_sim, top_items_sim_list, "Dynamic Prices (Simulation Data - Top 5)", OPEN_HOUR_START, OPEN_HOUR_END)
                    else:
                         st.warning("No top items found in the uploaded data to plot.")

                else:
                    st.warning("Could not calculate dynamic pricing based on the uploaded data and limits.")

            else:
                st.error("The uploaded file could not be processed or contains no valid data.")
        else:
            st.info("Please upload an Excel file to start the simulation.")


elif df_processed is None:
    st.error("Initial data file could not be loaded or processed. Cannot display analysis.")
    st.info(f"Ensure the file '{FILE_PATH}' exists in the same directory as the script and contains a sheet named '{SHEET_NAME}' with the required columns.")