import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# Custom page config
st.set_page_config(page_title='Water Quality Estimator', page_icon='💧', layout='centered')

# Sidebar for info and navigation
with st.sidebar:
    st.title('💧 Water Quality Estimator')
    st.markdown('''
    **Instructions:**
    - Enter the water quality parameters below.
    - Click **Predict** to see if the water is potable.
    - Adjust values as suggested to improve potability.
    ''')
    st.info('Tip: Watch the hints beside each field!')
    st.markdown('---')
    st.markdown('Created by:-')
    st.markdown('Piyush Thakur')
    st.markdown('Ankush Patial')

st.markdown('<h1 style="color:#0077b6;text-align:center;">Water Quality Estimator 💧</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Predict if water is potable based on its chemical properties.</p>', unsafe_allow_html=True)

# Load the trained Random Forest model
model = joblib.load('water_quality_rf_model.pkl')

# Typical potable ranges (can be adjusted based on data)
potable_ranges = {
    'PH': (6.5, 8.5),
    'Hardness': (60, 180),
    'Solids': (0, 12000),
    'Chloramines': (0, 8),
    'Sulfate': (0, 400),
    'Conductivity': (0, 600),
    'Trihalomethanes': (0, 80),
    'Turbidity': (0, 5)
}

# Helper for real-time suggestions
def get_hint(feature, value):
    low, high = potable_ranges[feature]
    if value < low:
        return f'⬆️ Increase to at least {low}'
    elif value > high:
        return f'⬇️ Decrease to at most {high}'
    else:
        return '✅ Good'

# Input fields in columns for better layout
col1, col2, col3 = st.columns(3)
with col1:
    PH = st.number_input('PH', min_value=0.0, max_value=14.0, value=7.0, help='Acidity/alkalinity (6.5-8.5 is safe)')
    st.caption(get_hint('PH', PH))
    hardness = st.number_input('Hardness', min_value=0.0, value=120.0, help='Calcium & magnesium (mg/L)')
    st.caption(get_hint('Hardness', hardness))
    solids = st.number_input('Solids', min_value=0.0, value=10000.0, help='Total dissolved solids (mg/L)')
    st.caption(get_hint('Solids', solids))
with col2:
    chloramines = st.number_input('Chloramines', min_value=0.0, value=7.0, help='Disinfectant (mg/L)')
    st.caption(get_hint('Chloramines', chloramines))
    sulfate = st.number_input('Sulfate', min_value=0.0, value=300.0, help='Sulfate ions (mg/L)')
    st.caption(get_hint('Sulfate', sulfate))
    conductivity = st.number_input('Conductivity', min_value=0.0, value=400.0, help='Electrical conductivity (μS/cm)')
    st.caption(get_hint('Conductivity', conductivity))
with col3:
    trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0, value=60.0, help='Byproducts (μg/L)')
    st.caption(get_hint('Trihalomethanes', trihalomethanes))
    turbidity = st.number_input('Turbidity', min_value=0.0, value=3.0, help='Cloudiness (NTU)')
    st.caption(get_hint('Turbidity', turbidity))

st.markdown('---')

# Prepare features and values for visualization
features = list(potable_ranges.keys())
input_values = [PH, hardness, solids, chloramines, sulfate, conductivity, trihalomethanes, turbidity]
min_values = [potable_ranges[f][0] for f in features]
max_values = [potable_ranges[f][1] for f in features]

# Parallel coordinates plot visualization
st.subheader("🔎 Visualize Your Water Quality Parameters")

# Prepare data for parallel coordinates
plot_df = pd.DataFrame({
    'Parameter': features,
    'Input': input_values,
    'Potable Min': min_values,
    'Potable Max': max_values
})

# Prepare data for parallel coordinates (each row is a line)
parcoords_df = pd.DataFrame([
    input_values,   # User input
    min_values,     # Potable min
    max_values      # Potable max
], columns=features)

# Assign colors for each line
line_colors = ['blue', 'green', 'red']
line_labels = ['Your Input', 'Potable Min', 'Potable Max']

# Create parallel coordinates plot (single trace, multiple lines)
fig = go.Figure(
    data=go.Parcoords(
        line=dict(color=[0, 1, 2], colorscale=[[0, 'blue'], [0.5, 'green'], [1, 'red']], showscale=False),
        dimensions=[
            dict(
                range=[potable_ranges[f][0], potable_ranges[f][1]],
                label=f,
                values=parcoords_df[f]
            ) for f in features
        ]
    )
)
fig.update_layout(
    height=500,
    margin=dict(l=50, r=50, t=50, b=50),
    plot_bgcolor='white',
    paper_bgcolor='white',
)

# Add a legend manually
st.markdown(
    '<span style="color:blue; font-weight:bold;">■</span> Your Input &nbsp; '
    '<span style="color:green; font-weight:bold;">■</span> Potable Min &nbsp; '
    '<span style="color:red; font-weight:bold;">■</span> Potable Max',
    unsafe_allow_html=True
)

st.plotly_chart(fig, use_container_width=True)

colA, colB = st.columns([2,1])
with colA:
    if st.button('🚰 Predict', use_container_width=True):
        # Prepare input data as a DataFrame
        data = {
            'PH': PH,
            'Hardness': hardness,
            'Solids': solids,
            'Chloramines': chloramines,
            'Sulfate': sulfate,
            'Conductivity': conductivity,
            'Trihalomethanes': trihalomethanes,
            'Turbidity': turbidity
        }
        df = pd.DataFrame([data])
        # Predict potability
        result = model.predict(df)
        prob = model.predict_proba(df)[0][1]
        proba= prob*100
        st.markdown(f'<h4>Probability of being Potable: <span style="color:#009688;">{proba:.2f}</span> %</h4>', unsafe_allow_html=True)
        if result[0] == 1:
            st.success('✅ The water is predicted to be **Potable**!')
        else:
            st.error('❌ The water is predicted to be **Not Potable**.')

with colB:
    if st.button('💡 Show Example Potable Values', use_container_width=True):
        # Load the original dataset (must be present in the same directory)
        df_data = pd.read_csv('water_potability.csv')
        potable_row = df_data[df_data['Potability'] == 1].iloc[0]
        st.markdown('**Example Potable Feature Values:**')
        st.json({
            'PH': potable_row['PH'],
            'Hardness': potable_row['Hardness'],
            'Solids': potable_row['Solids'],
            'Chloramines': potable_row['Chloramines'],
            'Sulfate': potable_row['Sulfate'],
            'Conductivity': potable_row['Conductivity'],
            'Trihalomethanes': potable_row['Trihalomethanes'],
            'Turbidity': potable_row['Turbidity']
        })
