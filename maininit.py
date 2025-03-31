import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import time
from sklearn.preprocessing import MinMaxScaler # For normalization

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Dynamic Pricing System")
st.title("🛒 Dynamic Pricing and Sales Analysis System")

# --- Constants & Assumptions ---
FILE_PATH = "dma_data.xlsx" # Make sure your Excel file name matches
SHEET_NAME = "Sheet1"
ASSUMED_COST_FACTOR = 0.60 # Assume cost is 60% of price. CHANGE IF YOU HAVE REAL DATA!
MIN_PRICE_CHANGE = 0.05 # Minimum % change for dynamic pricing
MAX_PRICE_CHANGE = 0.25 # Maximum % change for dynamic pricing
OPEN_HOUR_START = 8
OPEN_HOUR_END = 21 # Shop closes *before* 9 PM (exclusive)

# --- Caching Data Loading and Processing ---
@st.cache_data # Cache the data loading and initial processing
def load_and_process_data(file_path, sheet_name, cost_factor):
    try:
        xls = pd.ExcelFile(file_path)
        df_raw = pd.read_excel(xls, sheet_name=sheet_name, dtype={"Time_of_Purchase": str}) # Keep time as string initially
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Make sure '{file_path}' is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

    # Verify required columns exist
    required_cols = ["Purchased_Items", "Quantities", "Prices_per_Unit", "Time_of_Purchase"]
    if not all(col in df_raw.columns for col in required_cols):
        st.error(f"Missing one or more required columns in '{sheet_name}'. Expected: {', '.join(required_cols)}")
        return None

    # Clean and structure the data
    structured_data = []
    for index, row in df_raw.iterrows(): # Use index for better error reporting
        # Get data, handling potential missing values (NaN converted to string)
        items_str = str(row.get("Purchased_Items", ""))
        quantities_str = str(row.get("Quantities", ""))
        prices_str = str(row.get("Prices_per_Unit", ""))
        time_str = str(row.get("Time_of_Purchase", ""))
        transaction_id = row.get("Transaction_ID", f"Row_{index+2}") # Use Transaction_ID or Row number for logging

        # Split and strip, removing empty strings (handles trailing commas)
        items = [i.strip() for i in items_str.split(',') if i.strip()]
        quantities = [q.strip() for q in quantities_str.split(',') if q.strip()]
        prices = [p.strip() for p in prices_str.split(',') if p.strip()]

        # Check for length consistency
        len_items = len(items)
        len_qty = len(quantities)
        len_price = len(prices)

        if not (len_items == len_qty == len_price):
            st.warning(f"Data mismatch in {transaction_id}: {len_items} items, {len_qty} quantities, {len_price} prices. Skipping this transaction.")
            continue

        # Process time first for the entire transaction
        try:
            # Let pandas try to infer the format first, then try specific ones if needed
            purchase_time_obj = pd.to_datetime(time_str, errors='coerce').time()
            if pd.isna(purchase_time_obj):
                 # Try common formats if inference fails
                 try:
                     purchase_time_obj = pd.to_datetime(time_str, format='%H:%M', errors='coerce').time()
                 except ValueError:
                      try:
                          purchase_time_obj = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce').time()
                      except ValueError:
                          purchase_time_obj = pd.NaT # Mark as Not a Time if all parsing fails

            if pd.isna(purchase_time_obj):
                st.warning(f"Invalid time format '{time_str}' in {transaction_id}. Skipping transaction.")
                continue
            hour = purchase_time_obj.hour
        except Exception as e:
            st.warning(f"Error processing time '{time_str}' in {transaction_id}: {e}. Skipping transaction.")
            continue

        # Process each item in the transaction
        for i in range(len_items):
            try:
                item = items[i]
                quantity = float(quantities[i])
                price = float(prices[i])

                # Validate quantity and price
                if quantity <= 0 or price < 0:
                     st.warning(f"Invalid quantity ({quantity}) or price ({price}) for item '{item}' in {transaction_id}. Skipping item.")
                     continue

                cost = price * cost_factor # Assumed cost
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
                # Skip item if quantity or price is not numeric after stripping
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

@st.cache_data
def get_hourly_sales(df):
    """Aggregates sales quantity by hour and item."""
    if df is None or 'Hour' not in df.columns or 'Item' not in df.columns or 'Quantity' not in df.columns:
        return pd.DataFrame(columns=['Hour', 'Item', 'Quantity']) # Return empty DataFrame

    all_hours_items = pd.MultiIndex.from_product(
        [df['Item'].unique(), range(24)], names=['Item', 'Hour']
    )
    hourly_analysis = df.groupby(['Item', 'Hour'])['Quantity'].sum().reindex(all_hours_items, fill_value=0).reset_index()
    return hourly_analysis

@st.cache_data
def get_item_stats(df_hourly, open_start, open_end): # Added open_start, open_end
    """
    Calculates total quantity and average/std dev of hourly sales
    during OPEN hours for items.
    """
    if df_hourly.empty or 'Item' not in df_hourly.columns or 'Quantity' not in df_hourly.columns:
         # Return DF with all expected columns to avoid downstream errors
         return pd.DataFrame(columns=['Item', 'TotalQuantity', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen'])

    # Calculate total quantity across all hours
    total_sales = df_hourly.groupby('Item')['Quantity'].sum().reset_index().rename(columns={'Quantity':'TotalQuantity'})

    # Filter for open hours to calculate average and std dev
    open_hours_sales = df_hourly[(df_hourly['Hour'] >= open_start) & (df_hourly['Hour'] < open_end)]

    if open_hours_sales.empty:
        st.warning("No sales data found during open hours to calculate item statistics (Avg/StdDev).")
        # Create stats DF with 0s/NaNs but correct columns
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
        # Fill NaN StdDev (e.g., item sold only in one open hour) with 0
        open_stats['StdDevHourlyQtyOpen'] = open_stats['StdDevHourlyQtyOpen'].fillna(0)

    # Merge total sales with open hour stats
    item_performance = pd.merge(total_sales, open_stats, on='Item', how='left')
    # Fill stats for items potentially never sold during open hours
    item_performance['AvgHourlyQtyOpen'] = item_performance['AvgHourlyQtyOpen'].fillna(0.0)
    item_performance['StdDevHourlyQtyOpen'] = item_performance['StdDevHourlyQtyOpen'].fillna(0.0)

    # Keep the SalesStdDev calculated only on open hours for categorization consistency
    item_performance['SalesStdDev'] = item_performance['StdDevHourlyQtyOpen']


    return item_performance[['Item', 'TotalQuantity', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen', 'SalesStdDev']] # Ensure all columns are returned

@st.cache_data
def categorize_items(item_stats):
    """Categorizes items based on sales volume and consistency using quantiles."""
    if item_stats.empty:
        return item_stats

    # Define quantiles for categorization
    qty_median = item_stats['TotalQuantity'].quantile(0.5)
    std_median = item_stats['SalesStdDev'].quantile(0.5) # Use median for std dev too

    conditions = [
        (item_stats['TotalQuantity'] >= qty_median) & (item_stats['SalesStdDev'] >= std_median), # High Qty, High Std -> Potential Stars/Erratic
        (item_stats['TotalQuantity'] >= qty_median) & (item_stats['SalesStdDev'] < std_median),  # High Qty, Low Std -> Stars/Cash Cows
        (item_stats['TotalQuantity'] < qty_median) & (item_stats['SalesStdDev'] >= std_median),   # Low Qty, High Std -> Question Marks
        (item_stats['TotalQuantity'] < qty_median) & (item_stats['SalesStdDev'] < std_median)    # Low Qty, Low Std -> Dogs/Stable Low
    ]
    # More descriptive categories
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
        return pd.DataFrame(), pd.DataFrame() # Return empty DFs

    max_increase_factor = max_increase_pct / 100.0
    max_decrease_factor = max_decrease_pct / 100.0 # Keep as positive for calculation

    # --- Base Price Calculation (Median) ---
    base_prices = df.groupby('Item')['Price'].median().reset_index()
    base_prices.rename(columns={'Price': 'BasePrice'}, inplace=True)

    # --- Calculate Hourly Stats ONLY during OPEN hours ---
    open_hours_sales = df_hourly[(df_hourly['Hour'] >= open_start) & (df_hourly['Hour'] < open_end)]

    if open_hours_sales.empty:
        st.warning("No sales data found during specified open hours. Cannot calculate demand stats or dynamic prices.")
        # Create empty DFs with expected columns
        item_hourly_stats = pd.DataFrame(columns=['Item', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen'])
        dynamic_pricing_final = pd.DataFrame(columns=['Item', 'Hour', 'Status', 'BasePrice',
                                                     'DemandLevel (Qty)', 'DemandZScore', 'Adjustment', 'NewPrice'])
        return dynamic_pricing_final, item_hourly_stats

    item_hourly_stats = open_hours_sales.groupby('Item')['Quantity'].agg(
        AvgHourlyQtyOpen='mean',
        StdDevHourlyQtyOpen='std'
    ).reset_index()
    # Handle cases with zero standard deviation (e.g., sold only once or same qty always)
    item_hourly_stats['StdDevHourlyQtyOpen'] = item_hourly_stats['StdDevHourlyQtyOpen'].fillna(0)


    # --- Generate Pricing Table ---
    pricing_data = []
    all_items = df['Item'].unique() # Use all items from the input df
    hours_range = range(24)

    # Merge stats back to hourly sales data for easier lookup
    hourly_sales_with_stats = pd.merge(df_hourly, item_hourly_stats, on='Item', how='left')
    # Fill NaN stats for items that might exist outside open hours or have no variance
    hourly_sales_with_stats['AvgHourlyQtyOpen'] = hourly_sales_with_stats['AvgHourlyQtyOpen'].fillna(0)
    hourly_sales_with_stats['StdDevHourlyQtyOpen'] = hourly_sales_with_stats['StdDevHourlyQtyOpen'].fillna(0)

    # Create a lookup dictionary for faster access inside the loop
    stats_lookup = hourly_sales_with_stats.set_index(['Item', 'Hour']).to_dict('index')

    for item in all_items:
        if item not in base_prices['Item'].values:
            # st.warning(f"Skipping item '{item}' in pricing as it has no base price calculated.") # Can be noisy
            continue
        base_price = base_prices.loc[base_prices['Item'] == item, 'BasePrice'].iloc[0]

        for hour in hours_range:
            is_open = open_start <= hour < open_end
            status = "Open" if is_open else "Closed"
            adjustment_pct = 0.0
            z_score = np.nan
            current_qty = 0 # Default quantity

            lookup_key = (item, hour)
            hour_data = stats_lookup.get(lookup_key) # Use .get for safety

            if hour_data:
                 current_qty = hour_data.get('Quantity', 0)
                 avg_qty = hour_data.get('AvgHourlyQtyOpen', 0)
                 std_dev = hour_data.get('StdDevHourlyQtyOpen', 0)

                 if is_open:
                     # Calculate Z-score only if std_dev is not zero
                     if std_dev > 0:
                         z_score = (current_qty - avg_qty) / std_dev
                     elif avg_qty > 0: # If std_dev is 0 but there's an average
                         # Handle constant sales: if current is above avg (shouldn't happen if std=0), maybe small increase?
                         # If current is below avg (also shouldn't happen), maybe small decrease?
                         # For simplicity, let's treat std_dev=0 as neutral unless qty is also 0.
                         z_score = 0 # No deviation from the mean
                     # else: avg_qty is also 0, z_score remains NaN

                     # Apply adjustment based on Z-score using interpolation
                     # Map z-score range (e.g., -1.5 to 1.5) to adjustment range (-max_decrease to +max_increase)
                     z_min, z_max = -1.5, 1.5 # Define the z-score range for full adjustment scaling
                     adj_min, adj_max = -max_decrease_factor, max_increase_factor

                     if pd.notna(z_score):
                           # Interpolate: z_scores outside the range get clamped to min/max adjustment
                           adjustment_pct = np.interp(z_score, [z_min, z_max], [adj_min, adj_max])
                     # If z_score is NaN (e.g., avg=0, std=0), adjustment remains 0

            # No else needed: if not hour_data or not is_open, adjustment remains 0

            adjustment_pct = np.clip(adjustment_pct, -max_decrease_factor, max_increase_factor) # Ensure bounds
            new_price = round(base_price * (1 + adjustment_pct), 2)
            adj_str = f"{int(round(adjustment_pct * 100))}%" if is_open else "0% (Closed)"

            pricing_data.append({
                'Item': item,
                'Hour': f"{hour:02d}:00",
                'Status': status,
                'BasePrice': round(base_price, 2),
                'DemandLevel (Qty)': current_qty, # Show actual quantity for this hour
                'DemandZScore': round(z_score, 2) if pd.notna(z_score) else np.nan, # Show Z-score
                'Adjustment': adj_str,
                'NewPrice': new_price
            })

    dynamic_pricing_final = pd.DataFrame(pricing_data)

    # Merge Avg and Std Dev back for display purposes (optional but helpful)
    dynamic_pricing_final = pd.merge(dynamic_pricing_final, item_hourly_stats[['Item', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen']], on='Item', how='left')

    return dynamic_pricing_final, item_hourly_stats # Return stats too

@st.cache_data
def simulate_profitability(df, pricing_table): # pricing_table should be a DataFrame here
    """Simulates revenue and profit with static vs dynamic pricing."""
    # This check should now work correctly because pricing_table will be a DataFrame
    if df is None or df.empty or pricing_table.empty:
        return 0, 0, 0, 0

    # Merge dynamic prices back to the original transaction data
    pricing_map = pricing_table.set_index(['Item', 'Hour'])['NewPrice'].to_dict()
    base_price_map = pricing_table.set_index('Item')['BasePrice'].to_dict() # Assuming base price is constant per item

    total_static_revenue = 0
    total_static_profit = 0
    total_dynamic_revenue = 0
    total_dynamic_profit = 0

    # Use the original structured data (before grouping) for simulation
    for _, row in df.iterrows():
        item = row['Item']
        hour_int = row['Hour']
        hour_str = f"{hour_int:02d}:00"
        quantity = row['Quantity']
        cost = row['Cost'] # Use pre-calculated cost

        # Static Calculation
        base_price = base_price_map.get(item, 0) # Get base price for item
        static_revenue_txn = quantity * base_price
        static_profit_txn = quantity * (base_price - cost)
        total_static_revenue += static_revenue_txn
        total_static_profit += static_profit_txn

        # Dynamic Calculation
        dynamic_price_key = (item, hour_str)
        dynamic_price = pricing_map.get(dynamic_price_key, base_price) # Default to base price if not found/closed
        dynamic_revenue_txn = quantity * dynamic_price
        dynamic_profit_txn = quantity * (dynamic_price - cost)
        total_dynamic_revenue += dynamic_revenue_txn
        total_dynamic_profit += dynamic_profit_txn

    return total_static_revenue, total_static_profit, total_dynamic_revenue, total_dynamic_profit

# --- Plotting Functions ---

def plot_hourly_trend(df_hourly, items_to_plot, title, open_start, open_end, item_stats_sim=None): # Added item_stats_sim
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

        # --- Add average lines if stats are provided ---
        if item_stats_sim is not None and not item_stats_sim.empty:
            items_in_plot = plot_data['Item'].unique()
            stats_to_plot = item_stats_sim[item_stats_sim['Item'].isin(items_in_plot)]
            palette = sns.color_palette(n_colors=len(items_in_plot)) # Get colors used by seaborn
            item_color_map = dict(zip(items_in_plot, palette))

            for _, item_row in stats_to_plot.iterrows():
                item_name = item_row['Item']
                avg_qty = item_row['AvgHourlyQtyOpen']
                if item_name in item_color_map: # Check if item is plotted
                    ax.axhline(y=avg_qty, color=item_color_map[item_name], linestyle=':',
                               label=f'{item_name} Avg (Open Hrs)')
        # --- End average lines ---
    else:
        st.warning(f"No sales recorded during open hours ({open_start:02d}:00 - {open_end:02d}:00) for selected items.")
        ax.set_ylabel("Total Quantity Sold")

    hours_range = range(24)
    for h in hours_range:
        if not (open_start <= h < open_end):
            ax.axvspan(h - 0.5, h + 0.5, color='gray', alpha=0.15, zorder=0)

    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

    ax.set_xticklabels([f"{h:02d}:00" for h in hours_range], rotation=45, ha="right")
    ax.set_xlabel("Hour of Purchase (Full Day Context)")
    ax.set_ylabel("Total Quantity Sold")
    ax.set_title(title)

    # Adjust legend placement
    handles, labels = ax.get_legend_handles_labels()
    # Filter out potential duplicate average labels if using axhline label
    unique_labels = {}
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
            unique_handles.append(handle)

    ax.legend(unique_handles, unique_labels.keys(), title="Item / Avg", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, axis='y', linestyle='--')
    st.pyplot(fig)

def plot_sales_heatmap(df_hourly, top_n=15, open_start=0, open_end=24): # Added open hours params (with defaults)
    """Plots a heatmap of hourly sales for top N items, with lines for open hours."""
    if df_hourly.empty:
         st.warning("No data available for plotting heatmap.")
         return

    top_items = df_hourly.groupby('Item')['Quantity'].sum().nlargest(top_n).index
    heatmap_data = df_hourly[df_hourly['Item'].isin(top_items)]

    if heatmap_data.empty:
        st.warning("No data found for the top items to generate heatmap.")
        return

    # Ensure all 24 hours are columns, even if no sales, for consistent heatmap
    heatmap_pivot = heatmap_data.pivot_table(index='Item', columns='Hour', values='Quantity', fill_value=0)
    heatmap_pivot = heatmap_pivot.reindex(columns=range(24), fill_value=0) # Ensure all columns 0-23 exist

    fig, ax = plt.subplots(figsize=(15, max(6, len(top_items)*0.4))) # Adjust height
    sns.heatmap(heatmap_pivot, cmap="viridis", linewidths=.5, ax=ax, annot=False, cbar_kws={'label': 'Quantity Sold'})

    # --- Add vertical lines to indicate open/close times ---
    # Line before the start hour (e.g., at 8)
    ax.axvline(x=open_start, color='white', linestyle='--', linewidth=2)
    # Line after the last open hour (e.g., at 21, marking the end of hour 20)
    ax.axvline(x=open_end, color='white', linestyle='--', linewidth=2)
    # --- End lines ---

    ax.set_title(f'Hourly Sales Heatmap (Top {top_n} Items)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Item')
    ax.set_xticks(np.arange(24) + 0.5) # Center ticks for heatmap
    ax.set_xticklabels(range(24))
    plt.yticks(rotation=0)
    st.pyplot(fig)

def plot_item_categorization(item_stats):
    """Plots the item categorization scatter plot."""
    if item_stats.empty:
         st.warning("No data available for plotting item categories.")
         return

    # Optional: Normalize for better visualization if scales differ greatly
    scaler = MinMaxScaler()
    stats_scaled = item_stats.copy()
    # Avoid scaling 'Item' and 'Category' columns
    cols_to_scale = ['TotalQuantity', 'SalesStdDev']
    if all(col in stats_scaled.columns for col in cols_to_scale):
         stats_scaled[cols_to_scale] = scaler.fit_transform(stats_scaled[cols_to_scale])
         x_col, y_col = 'TotalQuantity', 'SalesStdDev'
         x_label, y_label = "Normalized Total Quantity", "Normalized Sales Std Dev"
    else:
         # Fallback if columns not present (shouldn't happen with current logic)
         st.warning("Required columns for scaling not found.")
         return


    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=stats_scaled, x=x_col, y=y_col, hue='Category', style='Category', s=100, ax=ax)
    ax.set_title('Item Performance Categorization')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--')

    # Optional: Add annotations for a few points if not too cluttered
    # for i, point in stats_scaled.iterrows():
    #    if i < 10: # Annotate first 10 for example
    #        ax.text(point[x_col]+0.01, point[y_col], str(point['Item']), fontsize=8)

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
    # *** FIX: Use 'DemandZScore' instead of 'DemandLevel (Normalized)' ***
    demand_zscores = item_pricing['DemandZScore'].tolist()
    # ********************************************************************
    base_price = item_pricing['BasePrice'].iloc[0]

    # --- Filter data for open hours ONLY for plotting lines ---
    open_hours_indices = [i for i, h in enumerate(hours_int) if open_start <= h < open_end]
    open_hours_int = [hours_int[i] for i in open_hours_indices]
    open_prices = [prices[i] for i in open_hours_indices]
    # Filter Z-scores, only keeping non-NaN values for plotting
    open_zscores = [demand_zscores[i] for i in open_hours_indices if pd.notna(demand_zscores[i])]
    # Get corresponding hours for valid z-scores
    open_hours_zscore_plot = [h for h, z in zip(open_hours_int, [demand_zscores[i] for i in open_hours_indices]) if pd.notna(z)]
    # --- End filtering ---


    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot Z-Score on primary axis (only open, valid points)
    color = 'tab:blue'
    ax1.set_xlabel('Hour of Day (Full Day Context)')
    # *** FIX: Update Y-axis label ***
    ax1.set_ylabel('Demand Z-Score', color=color)
    if open_hours_zscore_plot: # Check if there's z-score data to plot
        ax1.plot(open_hours_zscore_plot, open_zscores, color=color, marker='o', linestyle='--', label='Demand Z-Score')
        # Add a horizontal line at Z=0 for reference (average demand)
        ax1.axhline(y=0, color=color, linestyle=':', alpha=0.5, label='Avg Demand (Z=0)')
    ax1.tick_params(axis='y', labelcolor=color)
    # Adjust ylim for z-score, e.g., from -3 to 3 or based on data range
    if open_zscores:
         min_z, max_z = min(open_zscores), max(open_zscores)
         padding = max(1, (max_z - min_z) * 0.1) # Add padding or at least 1 unit
         ax1.set_ylim(min_z - padding, max_z + padding)
    else:
         ax1.set_ylim(-2, 2) # Default if no data


    # Plot Price on secondary axis (only open hour points)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Price ($)', color=color)
    if open_hours_int: # Check if there's price data to plot
        ax2.plot(open_hours_int, open_prices, color=color, marker='x', linestyle='-', label='Dynamic Price')
    ax2.axhline(y=base_price, color='grey', linestyle=':', label=f'Base Price (${base_price:.2f})')
    ax2.tick_params(axis='y', labelcolor=color)

    # Highlight open/closed periods visually on the full axis
    hours_range = range(24)
    for h in hours_range:
        if not (open_start <= h < open_end):
            ax1.axvspan(h - 0.5, h + 0.5, color='gray', alpha=0.15, zorder=0)

    ax1.set_xticks(range(24))
    ax1.set_xticklabels([f"{h:02d}:00" for h in hours_range], rotation=45, ha="right")
    ax1.set_xlim(-0.5, 23.5) # Ensure full 24 hours are shown
    ax1.grid(True, axis='y', linestyle=':')
    ax2.grid(False)

    fig.suptitle(f'Dynamic Pricing Strategy for {item} (Shop Hours: {open_start:02d}:00-{open_end:02d}:00)')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.93, 0.95)) # Adjust anchor slightly

    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
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

    # Convert Hour string ('HH:00') back to numeric for plotting
    filtered_pricing['HourInt'] = filtered_pricing['Hour'].apply(lambda x: int(x.split(':')[0]))

    # Filter data to include ONLY open hours for the line plot
    plot_data = filtered_pricing[(filtered_pricing['HourInt'] >= open_start) & (filtered_pricing['HourInt'] < open_end)]

    fig, ax = plt.subplots(figsize=(12, 6))

    if not plot_data.empty:
         sns.lineplot(data=plot_data, x="HourInt", y="NewPrice", hue="Item", marker="o", ax=ax)
         # Add base price lines for context, only need one per item
         base_prices_plot = plot_data.drop_duplicates(subset=['Item'])
         palette = sns.color_palette(n_colors=len(items_to_plot)) # Get colors used by seaborn
         item_color_map = dict(zip(base_prices_plot['Item'], palette))
         for _, item_row in base_prices_plot.iterrows():
             item_name = item_row['Item']
             base_price_val = item_row['BasePrice']
             if item_name in item_color_map:
                  ax.axhline(y=base_price_val, color=item_color_map[item_name], linestyle=':', label=f'{item_name} Base')
    else:
         st.warning(f"No dynamic pricing calculated during open hours ({open_start:02d}:00 - {open_end:02d}:00) for selected items.")
         ax.set_ylabel("Calculated Dynamic Price ($)") # Keep label


    # --- Highlight closed hours ---
    hours_range = range(24)
    for h in hours_range:
        if not (open_start <= h < open_end):
            ax.axvspan(h - 0.5, h + 0.5, color='gray', alpha=0.15, zorder=0)
    # --- End highlighting ---

    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}:00" for h in hours_range], rotation=45, ha="right")
    ax.set_xlim(-0.5, 23.5)
    # Adjust y-axis limits if needed, maybe based on min/max price plotted
    if not plot_data.empty:
         min_price = plot_data['NewPrice'].min()
         max_price = plot_data['NewPrice'].max()
         ax.set_ylim(bottom=min(0, min_price * 0.9), top=max_price * 1.1) # Add some padding
    else:
         ax.set_ylim(bottom=0) # Default if no data


    ax.set_xlabel("Hour of Day (Full Day Context)")
    ax.set_ylabel("Calculated Dynamic Price ($)") # Set again
    ax.set_title(title)

    # Adjust legend placement
    handles, labels = ax.get_legend_handles_labels()
    # Filter out potential duplicate average labels if using axhline label
    unique_labels = {}
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels.keys(), title="Item / Base", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.grid(True, axis='y', linestyle='--')
    st.pyplot(fig)

# --- Load Data ---
df_processed = load_and_process_data(FILE_PATH, SHEET_NAME, ASSUMED_COST_FACTOR)

# --- Main App Logic ---
if df_processed is not None:
    # --- Perform BASE Calculations (using cached functions on original data) ---
    hourly_sales_base = get_hourly_sales(df_processed)
    item_stats_base = get_item_stats(hourly_sales_base, OPEN_HOUR_START, OPEN_HOUR_END)
    item_categories_base = categorize_items(item_stats_base) # categorize uses 'SalesStdDev' which is now based on open hours std dev

    # *** CORRECTED: Unpack the tuple returned by calculate_dynamic_pricing ***
    # We only need the pricing table for the base simulation and base display, ignore the base stats for now.
    dynamic_pricing_base, _ = calculate_dynamic_pricing(
        df_processed, hourly_sales_base, MIN_PRICE_CHANGE, MAX_PRICE_CHANGE, OPEN_HOUR_START, OPEN_HOUR_END
    )
    # *************************************************************************

    # Now dynamic_pricing_base is the DataFrame, so this call is correct
    sim_results_base = simulate_profitability(df_processed, dynamic_pricing_base)

    # --- Sidebar ---
    st.sidebar.header("Filters & Options (Main Analysis)")
    # Select top N items for display (based on original data)
    all_items_sorted = item_stats_base.sort_values("TotalQuantity", ascending=False)['Item'].tolist()
    # Ensure max_value is not greater than the number of items available
    max_slider_value = min(20, len(all_items_sorted))
    if max_slider_value < 1: # Handle case with no items
        top_n_items = 0
        selected_top_items = []
        item_for_detail = None
        st.sidebar.warning("No item data for main analysis.")
    else:
        top_n_items = st.sidebar.slider(
            "Select No. of Top Items (Main):",
            min_value=1,
            max_value=max_slider_value,
            value=min(5, max_slider_value), # Default to 5 or max available if less
            key="slider_main" # Add unique key
        )
        selected_top_items = all_items_sorted[:top_n_items]

        # Select specific item for detailed view
        item_for_detail = st.sidebar.selectbox(
            "Select Item for Detailed View (Main):",
            all_items_sorted,
            key="select_main" # Add unique key
        )

    st.sidebar.info(f"Shop Operating Hours: {OPEN_HOUR_START:02d}:00 - {OPEN_HOUR_END:02d}:00")
    st.sidebar.warning(f"Cost Assumption: {int(ASSUMED_COST_FACTOR*100)}% of selling price.")

    # --- Tabs for Organization ---
    tab_list = [
        "📊 Sales Overview",
        "🏷️ Item Deep Dive",
        "📈 Dynamic Pricing",
        "⭐ Item Categories",
        "💰 Profit Simulation",
        "🚀 Simulation (Beta)" # Add new tab name
    ]
    tabs = st.tabs(tab_list)

    # Assign tabs to variables
    tab_overview = tabs[0]
    tab_deep_dive = tabs[1]
    tab_dynamic_pricing = tabs[2]
    tab_categories = tabs[3]
    tab_profit_sim = tabs[4]
    tab_simulation = tabs[5] # New simulation tab


    with tab_overview:
        st.header("Sales Overview (Based on Original Data)")

        st.subheader(f"Hourly Sales Trend (Top {top_n_items} Items)")
        if selected_top_items:
            # Pass base stats (item_stats_base) for average line plotting
            plot_hourly_trend(hourly_sales_base, selected_top_items, f"Hourly Sales Trend for Top {top_n_items} Items", OPEN_HOUR_START, OPEN_HOUR_END, item_stats_base)
        else:
             st.write("No items selected or available.")


        st.subheader("Overall Hourly Sales (All Items)")
        overall_hourly = df_processed.groupby('Hour')['Quantity'].sum().reset_index()
        if not overall_hourly.empty:
             fig_overall, ax_overall = plt.subplots(figsize=(12, 5))
             sns.lineplot(data=overall_hourly, x="Hour", y="Quantity", marker="o", ax=ax_overall)
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
             st.write("No overall hourly data available.")


        st.subheader(f"Hourly Sales Heatmap (Top {min(15, len(all_items_sorted))} Items)")
        if not hourly_sales_base.empty:
            plot_sales_heatmap(hourly_sales_base, top_n=min(15, len(all_items_sorted)), open_start=OPEN_HOUR_START, open_end=OPEN_HOUR_END)
        else:
             st.write("No hourly data available for heatmap.")


    with tab_deep_dive:
        st.header(f"Deep Dive (Original Data): {item_for_detail if item_for_detail else 'No Item Selected'}")

        if item_for_detail:
            st.subheader("Hourly Sales Trend")
            # Pass base stats for average line plotting
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
                st.metric(label="Sales Standard Deviation", value=f"{category_info['SalesStdDev'].iloc[0]:.2f}")
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
             if show_all_base:
                  # Display more informative columns
                  cols_to_show_base = ['Item', 'Hour', 'Status', 'BasePrice', 'DemandLevel (Qty)', 'DemandZScore', 'Adjustment', 'NewPrice', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen']
                  st.dataframe(dynamic_pricing_base[cols_to_show_base], height=500)
             elif selected_top_items:
                  cols_to_show_base = ['Item', 'Hour', 'Status', 'BasePrice', 'DemandLevel (Qty)', 'DemandZScore', 'Adjustment', 'NewPrice']
                  st.dataframe(dynamic_pricing_base[dynamic_pricing_base['Item'].isin(selected_top_items)][cols_to_show_base], height=500)
             else:
                 st.write("No top items selected.")

        else:
             st.warning("Dynamic pricing table (base) could not be generated.")


    with tab_categories:
        st.header("Item Performance Categories (Original Data)")
        st.write("Items categorized by total sales volume and sales consistency (standard deviation across hours).")
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


    # --- Simulation Tab ---
    with tab_simulation:
        st.header("🚀 Simulation Environment (Beta)")
        st.info("Upload an Excel file with the same format as the original data to run a simulation with custom dynamic pricing limits.")

        uploaded_file = st.file_uploader("Upload Simulation Data (.xlsx)", type="xlsx", key="sim_upload")

        st.markdown("---")
        st.subheader("Set Simulation Pricing Limits:")
        # Use separate sliders for increase and decrease percentages
        max_decrease_pct_sim = st.slider("Max Price Decrease (%)", min_value=0, max_value=50, value=int(MIN_PRICE_CHANGE*100), step=1, key="slider_sim_decrease") # Default to old MIN_CHANGE
        max_increase_pct_sim = st.slider("Max Price Increase (%)", min_value=0, max_value=50, value=int(MAX_PRICE_CHANGE*100), step=1, key="slider_sim_increase")

        if uploaded_file is not None:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")

            # Load and process the uploaded data
            df_sim = load_and_process_data(uploaded_file, SHEET_NAME, ASSUMED_COST_FACTOR)

            if df_sim is not None and not df_sim.empty:
                st.markdown("---")
                st.subheader("Simulation Results")

                # Recalculate based on uploaded data and simulation limits
                hourly_sales_sim = get_hourly_sales(df_sim)
                # Use the updated calculate_dynamic_pricing which returns stats too
                dynamic_pricing_sim, item_stats_sim = calculate_dynamic_pricing(
                    df_sim, hourly_sales_sim, max_increase_pct_sim, max_decrease_pct_sim, OPEN_HOUR_START, OPEN_HOUR_END
                )

                if not dynamic_pricing_sim.empty:
                    # Determine top items from the SIMULATION data
                    if not item_stats_sim.empty:
                         # Calculate total quantity for sorting
                         total_qty_sim = hourly_sales_sim.groupby('Item')['Quantity'].sum().reset_index().rename(columns={'Quantity':'TotalQuantity'})
                         item_stats_sim_merged = pd.merge(item_stats_sim, total_qty_sim, on='Item', how='left').fillna(0)
                         top_items_sim_list = item_stats_sim_merged.sort_values("TotalQuantity", ascending=False)['Item'].head(5).tolist()
                    else:
                         top_items_sim_list = []
                         st.warning("Could not determine demand stats or top items from simulation data.")


                    st.write(f"Dynamic Pricing Table (Using {max_decrease_pct_sim}% Decrease / {max_increase_pct_sim}% Increase limits)")
                    # Display more informative columns
                    cols_to_show_sim = ['Item', 'Hour', 'Status', 'BasePrice', 'DemandLevel (Qty)', 'DemandZScore', 'Adjustment', 'NewPrice', 'AvgHourlyQtyOpen', 'StdDevHourlyQtyOpen']
                    st.dataframe(dynamic_pricing_sim[cols_to_show_sim], height=400)


                    if top_items_sim_list:
                         st.write("Hourly Sales Frequency Graph (Top 5 from Uploaded Data)")
                         # Pass item_stats_sim to plot the average lines
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


# --- Fallback if initial data loading failed ---
elif df_processed is None:
    st.error("Initial data file could not be loaded or processed. Cannot display analysis.")
    st.info(f"Ensure the file '{FILE_PATH}' exists in the same directory as the script and contains a sheet named '{SHEET_NAME}' with the required columns.")