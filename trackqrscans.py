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
    if df.empty:
        st.warning("No records to display for the selected period.")
        return

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Format dates without the year
    df['DateFormatted'] = df['Date'].dt.strftime('%m/%d')

    # Set custom colors
    bar_color = '#E0A100'  # Gold color for bars
    text_color = 'white'   # White text for contrast

    # Increase figure size
    fig, ax = plt.subplots(figsize=(16, 9))  # Adjust figsize as needed

    # Set figure and axes background to transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Create the bar plot with custom bar color
    sns.barplot(x='DateFormatted', y='ScanCount', data=df, color=bar_color, ax=ax)

    # Customize the plot
    ax.set_xlabel('Date', fontsize=20, color=text_color)
    ax.set_ylabel('Scan Count', fontsize=20, color=text_color)
    ax.set_title('', fontsize=20, color=text_color)
    ax.tick_params(colors=text_color)
    plt.xticks(rotation=0, ha='right', color=text_color)
    plt.yticks(color=text_color)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Display the total number of scans
    plt.text(0.99, 0.99, f'Total Scans: {total_scans}', transform=ax.transAxes,
             ha='right', va='top', fontsize=40, color=text_color)

    plt.tight_layout()


    # Save the figure to a buffer
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    buf.seek(0)
    import base64
    img_base64 = base64.b64encode(buf.read()).decode()

    # Center the plot using HTML
    st.markdown(
        f'''
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{img_base64}" style="max-width: 170%; height: auto;">
        </div>
        ''',
        unsafe_allow_html=True
    )
    
def set_text_styles():
    st.markdown(
        """
        <style>
        /* Main content text */
        .stApp, .stApp * {
            color: white;
            background-color: transparent;
        }

        /* Sidebar background */
        .css-1d391kg {
            background-color: rgba(0, 0, 0, 0.5);
        }

        /* Sidebar text */
        .css-1d391kg h2, .css-1d391kg label {
            color: white;
        }

        /* Style the buttons */
        .stButton > button, .stDownloadButton > button {
            background-color: #022454;
            color: white;
            border-radius: 8px;
            height: 50px;
            width: 100%;
            font-size: 16px;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: #A17401;
            color: white;
        }

        /* Style date input widgets */
        .stDateInput input {
            background-color: #022454;
            color: white;
        }
        .stDateInput [role="button"] {
            background-color: #022454;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    
def set_bg_color(bg_color, sidebar_color):
    """
    Set the background color of the Streamlit app and sidebar.

    Args:
        bg_color (str): Hex code of the main background color.
        sidebar_color (str): Hex code of the sidebar background color.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
        }}
        .css-1d391kg {{
            background-color: {sidebar_color};
        }}
        .css-1d391kg, .css-1d391kg * {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def set_background_image(image_url):
    """
    Sets a background image with an overlay for the Streamlit app.

    Args:
        image_url (str): URL of the background image.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(3, 35, 90, 0.7),
                rgba(3, 35, 90, 0.7)
            ),
            url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def add_transparent_overlay():
    st.markdown(
        """
        <style>
        .overlay {
            position: fixed;
            top: 0;
            bottom: 0;
            right: 0;
            left: 0;
            background-color: rgba(0, 0, 90, 2); /* Adjust the alpha value for transparency */
            z-index: 1;
        }
        </style>
        <div class="overlay"></div>
        """,
        unsafe_allow_html=True
    )


def main():
    # Set background image with overlay
    background_image_url = 'https://calbaptist.edu/_resources/images/_news/2021-CBU-Campus-45.jpeg'
    set_background_image(background_image_url)

    # Set text styles
    set_text_styles()

    st.markdown(
        "<h1 style='color: white;'>QR Code Scan History:</h1>",
        unsafe_allow_html=True
    )

    # Fetch data and proceed as usual
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
