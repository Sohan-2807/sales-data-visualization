import os
import matplotlib.pyplot as plt
import pandas as pd

# Set global Matplotlib design styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Loads CSV sales data, cleans invalid dates and missing numbers,

    and derives financial columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source data file not found at: {file_path}")

    df = pd.read_csv(file_path)
    initial_rows = len(df)
    print(f"[1/6] Raw dataset loaded: {initial_rows} records.")

    # 1. Parse and sanitize dates (corrupted dates become NaT)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # 2. Ensure numeric datatypes for numerical columns
    numeric_cols = ["Quantity", "Unit_Price", "Unit_Cost"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Drop records with missing vital metrics or dates
    df.dropna(subset=["Date", "Quantity", "Unit_Price", "Unit_Cost", "Product", "Category"], inplace=True)
    df["Quantity"] = df["Quantity"].astype(int)

    cleaned_rows = len(df)
    print(f"[2/6] Data cleaning complete: {initial_rows - cleaned_rows} invalid record(s) discarded. Remaining: {cleaned_rows}")

    # 4. Feature engineering: Calculate financial aggregates
    df["Revenue"] = df["Quantity"] * df["Unit_Price"]
    df["Total_Cost"] = df["Quantity"] * df["Unit_Cost"]
    df["Profit"] = df["Revenue"] - df["Total_Cost"]

    # Extract Year-Month formatted for chronological sorting
    df["Year_Month"] = df["Date"].dt.to_period("M")
    df["Month_Name"] = df["Date"].dt.strftime("%b %Y")

    return df


def generate_visualizations(df: pd.DataFrame, output_dir: str = "charts"):
    """Calculates aggregates and renders all 4 required charts."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[3/6] Output directory ready at: ./{output_dir}/")

    # -------------------------------------------------------------
    # Chart 1: Line Chart - Monthly Revenue Trend
    # -------------------------------------------------------------
    print("[4/6] Generating Chart 1: Monthly Revenue Trend...")
    monthly_data = (
        df.groupby("Year_Month")
        .agg(Revenue=("Revenue", "sum"))
        .reset_index()
    )
    monthly_data["Display_Month"] = monthly_data["Year_Month"].dt.strftime("%b %Y")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(
        monthly_data["Display_Month"],
        monthly_data["Revenue"],
        marker="o",
        linewidth=2.5,
        markersize=6,
        color="#1f77b4",
        label="Monthly Revenue"
    )

    # Highlight peak revenue month
    peak_idx = monthly_data["Revenue"].idxmax()
    peak_val = monthly_data.loc[peak_idx, "Revenue"]
    peak_month = monthly_data.loc[peak_idx, "Display_Month"]
    ax1.annotate(
        f"Peak: ${peak_val:,.0f}",
        xy=(peak_idx, peak_val),
        xytext=(peak_idx - 0.5, peak_val + 700),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#d62728"
    )

    ax1.set_title("Monthly Revenue Pattern (2025)", fontsize=13, pad=15, fontweight="bold")
    ax1.set_xlabel("Month", labelpad=10, fontweight="semibold")
    ax1.set_ylabel("Revenue ($ USD)", labelpad=10, fontweight="semibold")
    ax1.yaxis.set_major_formatter("${x:,.0f}")
    ax1.set_ylim(0, monthly_data["Revenue"].max() * 1.15)
    plt.xticks(rotation=45, ha="right")
    ax1.legend(loc="upper left")
    plt.tight_layout()

    chart1_path = os.path.join(output_dir, "1_monthly_revenue_trend.png")
    fig1.savefig(chart1_path, dpi=300)
    plt.show()

    # -------------------------------------------------------------
    # Chart 2: Horizontal Bar Chart - Revenue by Product
    # -------------------------------------------------------------
    print("[4/6] Generating Chart 2: Revenue Comparison by Product...")
    product_revenue = (
        df.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=True)
    )

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars2 = ax2.barh(product_revenue.index, product_revenue.values, color="#2ca02c", edgecolor="#1e711e")

    # Annotate bar values
    for bar in bars2:
        width = bar.get_width()
        ax2.text(
            width + 400,
            bar.get_y() + bar.get_height() / 2,
            f"${width:,.0f}",
            va="center",
            ha="left",
            fontsize=9
        )

    ax2.set_title("Total Revenue by Product", fontsize=13, pad=15, fontweight="bold")
    ax2.set_xlabel("Total Revenue ($ USD)", labelpad=10, fontweight="semibold")
    ax2.set_ylabel("Product Name", labelpad=10, fontweight="semibold")
    ax2.xaxis.set_major_formatter("${x:,.0f}")
    ax2.set_xlim(0, product_revenue.max() * 1.15)
    plt.tight_layout()

    chart2_path = os.path.join(output_dir, "2_revenue_by_product.png")
    fig2.savefig(chart2_path, dpi=300)
    plt.show()

    # -------------------------------------------------------------
    # Chart 3: Pie Chart - Category Revenue Share
    # -------------------------------------------------------------
    print("[4/6] Generating Chart 3: Category Revenue Distribution...")
    category_revenue = (
        df.groupby("Category")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    fig3, ax3 = plt.subplots(figsize=(8, 7))
    palette = ["#4e79a7", "#f28e2c", "#e15759"]
    explode = [0.05 if i == 0 else 0 for i in range(len(category_revenue))]

    wedges, texts, autotexts = ax3.pie(
        category_revenue.values,
        labels=category_revenue.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=palette[:len(category_revenue)],
        explode=explode,
        wedgeprops=dict(width=0.85, edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=10)
    )

    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")

    ax3.set_title("Revenue Contribution by Category", fontsize=13, pad=20, fontweight="bold")
    plt.tight_layout()

    chart3_path = os.path.join(output_dir, "3_category_revenue_share.png")
    fig3.savefig(chart3_path, dpi=300)
    plt.show()

    # -------------------------------------------------------------
    # Chart 4: Vertical Bar Chart - Profit by Product
    # -------------------------------------------------------------
    print("[4/6] Generating Chart 4: Total Profit by Product...")
    product_profit = (
        df.groupby("Product")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    bars4 = ax4.bar(
        product_profit.index,
        product_profit.values,
        color="#ff7f0e",
        edgecolor="#b35504",
        width=0.55
    )

    for bar in bars4:
        height = bar.get_height()
        ax4.annotate(
            f"${height:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax4.set_title("Total Profit Comparison Across Products", fontsize=13, pad=15, fontweight="bold")
    ax4.set_xlabel("Product Name", labelpad=10, fontweight="semibold")
    ax4.set_ylabel("Profit ($ USD)", labelpad=10, fontweight="semibold")
    ax4.yaxis.set_major_formatter("${x:,.0f}")
    ax4.set_ylim(0, product_profit.max() * 1.15)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    chart4_path = os.path.join(output_dir, "4_profit_by_product.png")
    fig4.savefig(chart4_path, dpi=300)
    plt.show()

    print("[5/6] All 4 charts successfully rendered and saved to the 'charts/' folder.")


def print_summary_statistics(df: pd.DataFrame):
    """Outputs a text summary table for verification during viva demo."""
    total_rev = df["Revenue"].sum()
    total_cost = df["Total_Cost"].sum()
    net_profit = df["Profit"].sum()
    margin = (net_profit / total_rev) * 100 if total_rev else 0

    print("\n" + "=" * 50)
    print("         EXECUTIVE PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total Transactions Processed : {len(df):,}")
    print(f"Gross Revenue                : ${total_rev:,.2f}")
    print(f"Total Operational Cost       : ${total_cost:,.2f}")
    print(f"Net Profit                   : ${net_profit:,.2f}")
    print(f"Overall Profit Margin        : {margin:.1f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    DATA_FILE = "sales_data.csv"
    CHARTS_DIR = "charts"

    # Execute workflow
    sales_df = load_and_clean_data(DATA_FILE)
    generate_visualizations(sales_df, output_dir=CHARTS_DIR)
    print_summary_statistics(sales_df)
    print("[6/6] Execution finished successfully.")