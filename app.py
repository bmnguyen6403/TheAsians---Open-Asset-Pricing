import streamlit as st

# Set page title
st.title("My First Streamlit App")

# Add a header
st.header("Welcome to my website!")

# Add some text
st.write("This is a simple Streamlit app. Edit the code to build your own!")

# Add an interactive slider
number = st.slider("Select a number:", 0, 100, 50)
st.write(f"You selected: {number}")

# Add a button
if st.button("Click Me!"):
    st.success("Button clicked! 🎉")
