import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb



selected = option_menu(
        menu_title = None,
        options = ['Home', "Evaluate", "Explore"],
        icons = ["house", "book", "envelope"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal",
        
    )
    

html_code = """
<html>
  <head>
    <style>
      /* Add your CSS code here */
    </style>
  </head>
  <body>
    <h1>Hello Streamlit!</h1>
  </body>
</html>
"""

pickled_model = pickle.load(open('rfr_model.pkl', 'rb'))



streets = (
    'street_Academy', 'street_Adamasingba', 'street_Agbowo', 'street_Agodi',
       'street_Akala Estate', 'street_Akala express', 'street_Akobo',
       'street_Alakia', 'street_Alalubosa', 'street_Apata', 'street_Apete',
       'street_Basorun', 'street_Bodija', 'street_Carlton Gate',
       'street_Elewuro', 'street_Eleyele', 'street_Heritage Estate',
       'street_Idishin', 'street_Ikolaba', 'street_Ire Akari', 'street_Iwo Rd',
       'street_Iyaganku', 'street_Jericho', 'street_Kolapo Ishola',
       'street_Kuola', 'street_Lagelu Estate', 'street_Mokola',
       'street_Molete', 'street_Moniya', 'street_New Garage', 'street_Odo Ona',
       'street_Oje', 'street_Ojoo', 'street_Ojurin', 'street_Oke ado',
       'street_Ologuneru', 'street_Olorunsogo', 'street_Oluyole',
       'street_Onireke Gra', 'street_Oremeji', 'street_Orogun', 'street_Ring',
       'street_Samonda', 'street_Sanyo', 'street_Soka', 'street_Yemetu',
       'street_challenge', 'street_other'
        
)



others = (
    
    'semi_detached', 'luxury', 'pop',
       'water_heater', 'wardrobe', 
       'teraced', 'proximity',
       'estate', 'detached'
    
)

house_type = (
    
    'bungalow', 'flat', 'duplex',
    'self_contain', 'room_parlor'
    
)


demo_df = pd.read_csv('demodf.csv', index_col = 0)

column_names = demo_df.columns

# Create a new empty DataFrame with the same column names as the existing DataFrame
input_data = pd.DataFrame(columns=column_names)

def predict_df(df, bedroom, bathroom, toilet, *column_names):
    # Get the number of columns in the DataFrame
    num_columns = len(df.columns)
    
    # Create a new row of data with the value 0 in every column
    new_row = [0] * num_columns
    
    # Set the value to 1 for the specified columns
    for col in column_names:
        new_row[df.columns.get_loc(col)] = 1
    
    # Set the value of bedroom, bathroom, and toilet in the appropriate columns
    new_row[df.columns.get_loc('bedroom')] = bedroom
    new_row[df.columns.get_loc('bathroom')] = bathroom
    new_row[df.columns.get_loc('toilet')] = toilet
    
    # Insert the new row into the DataFrame
    df.loc[len(df)] = new_row
    
    return df


def show_predict_page():
    st.title("Ibadan House Rent Prediction")

    st.write(""" #### Some information here""")
    
    street = st.selectbox('Street', streets)
    types = st.selectbox('House_type', house_type)
    selected_values = st.multiselect(
        "Amenities:",
        ['semi_detached', 'luxury', 'pop',
        'water_heater', 'wardrobe', 
        'teraced', 'close to main road',
        'estate', 'detached','newly_built', 'furnished', 'serviced'],
        default=["wardrobe"])

    bedroom = st.slider('Bedrooms', 0, 10,1)
    bathroom = st.slider('Bathrooms', 0, 10,1)
    toilet = st.slider('Toilets', 0, 10,1)

    ok = st.button("Predict Rent")

    if ok:
        X = predict_df(input_data, bedroom,bathroom,toilet,types, street)
        loaded_regressor = pickled_model
        pred = loaded_regressor.predict(input_data)
        pred = np.exp(pred)
        
        st.subheader(f'The estimated rent is around {pred} naira')
        #st.subheader(selected_values)
    
    



if selected == 'Home':
    st.title(f'You have selected {selected}')
    #st.write(""" #### Some information here""")
    st.markdown(html_code, unsafe_allow_html=True)
if selected == 'Evaluate':
    st.title(f'You have selected {selected}')
    show_predict_page()
if selected == 'Explore':
    st.title(f'You have selected {selected}')
    st.write(""" #### Some information here""")