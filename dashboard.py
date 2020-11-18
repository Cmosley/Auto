# from collections import namedtuple
# import altair as alt
# import math
# import pandas as pd
# import streamlit as st
# import numpy as np


# progress_bar = st.progress(0)
# status_text = st.empty()
# chart = st.line_chart(np.random.randn(10, 2))

# for i in range(100):
#     # Update progress bar.
#     progress_bar.progress(i + 1)

#     new_rows = np.random.randn(10, 2)

#     # Update status text.
#     status_text.text(
#         'The latest random number is: %s' % new_rows[-1, 1])

#     # Append data to the chart.
#     chart.add_rows(new_rows)

#     # Pretend we're doing some computation that takes time.
#     time.sleep(0.1)

# status_text.text('Done!')
# st.balloons()


import streamlit as st
import pandas as pd

import streamlit.components.v1 as components

import warnings
warnings.filterwarnings('ignore')

@st.cache
def load_data1():
    dash = pd.read_csv("./datasets/Dashboard.csv")
    dash['Date'] = pd.to_datetime(dash['Date'])
    dash.set_index('Date', inplace=True)
    return dash

dash = load_data1()

def main():
    """Semi Automated ML App with Streamlit """

    activities = ["EDA","Plots"]	
    choice = st.sidebar.selectbox("Select Activities",activities)

    if choice == 'Plots':
        st.subheader("Data Visualization")
        
        all_columns_names = dash.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
        selected_columns_names = st.selectbox("Select Columns To Plot",all_columns_names)

        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

            # Plot By Streamlit
            if type_of_plot == 'area':
                cust_data = dash[selected_columns_names]
                st.area_chart(cust_data)

            elif type_of_plot == 'bar':
                cust_data = dash[selected_columns_names]
                st.bar_chart(cust_data)

            elif type_of_plot == 'line':
                cust_data = dash[selected_columns_names]
                st.line_chart(cust_data)

            # Custom Plot 
            elif type_of_plot:
                cust_plot= dash[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()
                    

if __name__ == '__main__':
    main()