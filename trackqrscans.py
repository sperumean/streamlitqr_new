import os
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pytz

# Fetch database credentials from environment variables
config = {
    'user': os.getenv('DB_USER', 'steven'),
    'password': os.getenv('DB_PASSWORD', 'Spiderman57#'),
    'host': os.getenv('DB_HOST', 'itsonlyfunifyoulive.duckdns.org'),
    'database': 'qrcode',
}

def fetch_scan_records_by_date(start_date=None, end_date=None):
    """
    Fetch scan records from the database, grouped by date.
    Optionally filter by date range.

    Args:
        start_date (datetime.date): The start date for data fetching.
        end_date (datetime.date): The end date for data fetching.

    Returns:
        df (DataFrame): Contains date and scan count.
        total_scans (int): Total number of scans.
    """
    try:
        # Establish a connection to the MySQL database
        with mysql.connector.connect(**config) as connection:
            # Create a cursor object to execute SQL queries
            with connection.cursor() as cursor:
                # Adjust the timestamp for time zone difference
                sql_timezone_adjustment = "DATE(CONVERT_TZ(timestamp, '+00:00', '-07:00'))"
                
                # SQL query to retrieve scan counts by date
                sql = f"""
                    SELECT {sql_timezone_adjustment} AS date,
                           COUNT(*) AS scan_count
                    FROM qr_scans
                """
                
                params = ()
                if start_date and end_date:
                    sql += f" WHERE {sql_timezone_adjustment} >= %s AND {sql_timezone_adjustment} <= %s"
                    params = (start_date, end_date)
                
                sql += f"""
                    GROUP BY {sql_timezone_adjustment}
                    ORDER BY {sql_timezone_adjustment}
                """
                
                # Execute the SQL query
                cursor.execute(sql, params)
                # Fetch all the scan records
                records = cursor.fetchall()
                
                # Create a DataFrame
                df = pd.DataFrame(records, columns=['Date', 'ScanCount'])
                total_scans = df['ScanCount'].sum()
                return df, total_scans
    except mysql.connector.Error as error:
        st.error(f"Error fetching scan records: {error}")
        return pd.DataFrame(), 0

def plot_scan_history_by_date(df, total_scans):
    """
    Plot a bar chart of the scan history by date.

    Args:
        df (DataFrame): Contains date and scan count.
        total_scans (int): Total number of scans.
    """
    if df.empty:
        st.warning("No records to display for the selected period.")
        return

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Format dates without the year
    df['DateFormatted'] = df['Date'].dt.strftime('%m/%d')

    # Set Seaborn style
    sns.set_style('whitegrid')

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='DateFormatted', y='ScanCount', data=df, color='skyblue', ax=ax)

    # Customize the plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Scan Count', fontsize=12)
    ax.set_title('QR Code Scan History by Date', fontsize=16)
    plt.xticks(rotation=45, ha='right')

    # Display the total number of scans
    plt.text(0.99, 0.99, f'Total Scans: {total_scans}', transform=ax.transAxes,
             ha='right', va='top', fontsize=12, color='black')

    plt.tight_layout()
    st.pyplot(fig)

def main():
    """
    Main function to fetch scan records and plot the scan history by date.
    """
    st.title("QR Code Scan History")

    # Fetch all data to get date range
    df_all, _ = fetch_scan_records_by_date()
    if df_all.empty:
        st.warning("No scan records available.")
        return

    # Sidebar for date selection
    st.sidebar.header("Filter Options")
    start_date = st.sidebar.date_input('Start Date', value=df_all['Date'].min())
    end_date = st.sidebar.date_input('End Date', value=df_all['Date'].max())

    if start_date > end_date:
        st.error("Error: Start Date must be before or equal to End Date.")
        return

    # Fetch data based on selected date range
    df, total_scans = fetch_scan_records_by_date(start_date, end_date)

    # Plot the scan history
    plot_scan_history_by_date(df, total_scans)

    # Download data as CSV
    csv = df.to_csv(index=False)
    st.download_button(label='Download Data as CSV', data=csv, file_name='scan_history.csv', mime='text/csv')

if __name__ == "__main__":
    main()
