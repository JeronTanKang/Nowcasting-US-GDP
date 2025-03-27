import streamlit as st

def display_dashboard(nowcast_df, model_type):
    st.title(f"{model_type} GDP Nowcasting Results")

    if nowcast_df.empty:
        st.error("No nowcast data to display.")
        return

    # GDP Growth Line Chart
    st.subheader("Nowcasted GDP Growth (%)")
    st.line_chart(data=nowcast_df.set_index("date")["Nowcasted_GDP_Growth"])

    # GDP Level Line Chart
    st.subheader("Nowcasted GDP Level")
    st.line_chart(data=nowcast_df.set_index("date")["Nowcasted_GDP"])

    # Expandable detailed results table
    with st.expander("See detailed nowcast results"):
        st.dataframe(nowcast_df.style.format({
            "Nowcasted_GDP_Growth": "{:.2f}",
            "Nowcasted_GDP": "{:,.2f}"
        }))
    
    # More UI elements like buttons, sliders, etc., can be added here
    st.write("Explore more features here!")

