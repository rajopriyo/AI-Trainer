import streamlit as st

st.set_page_config(
    page_title="AI Trainer",
    page_icon="💪",
)

# Title with deep green color
st.markdown("<h1 style='color: #006400;'>Choose Exercise to Track 😉💪</h1>", unsafe_allow_html=True)

# Sidebar and main content area styling
st.markdown(
    """
    <style>
        .css-18e3th9 {
            background-color: #000000 !important;
        }
        .css-1aumxhk {
            color: #f0f0f0;
        }
        .css-1lcbmhc, .css-1y4p8pa, .css-1lze3ba, .css-12ttj6m {
            background-color: #000000 !important;
        }
        .stApp {
            background-color: #000000;
            color: #f0f0f0;
        }
        h1 {
            color: #006400;
        }
        @media (min-width: 600px) {
            .cards-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.success("Select an Exercise Above")

def generate_card(title, description, links):
    card_html = f"""
        <div style="background-color: #1e1e1e; border: 2px solid #bcff96; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #bcff96;">{title}</h2>
            <p style="color: #f0f0f0;">{description}</p>
            <a href="{links}">
                <button style="padding: 10px 20px; background-color: #bcff96; color: #1e1e1e; border-radius: 5px; cursor: pointer; font-size: 1.2rem;">
                Start Exercise
                </button>
            </a>
        </div>
    """
    return card_html

st.markdown("<div class='cards-container'>", unsafe_allow_html=True)

# Generate and display multiple cards
card1 = generate_card("Hammer Curl", "Track your Hammer Curl exercise.", "https://webapplication.streamlit.app/HammerCurl")
card2 = generate_card("Jump", "Track your Jump exercise.", "https://ai-trainer-webapp.streamlit.app/Jump")
card3 = generate_card("Lifting", "Track your Lifting exercise.", "hhttps://ai-trainer-webapp.streamlit.app/Lifting")
card4 = generate_card("Pushup", "Track your Pushup exercise.", "https://ai-trainer-webapp.streamlit.app/Pushup")
card5 = generate_card("Indian Curl", "Track your Indian Curl exercise.","https://ai-trainer-webapp.streamlit.app/IndianCurl")
st.markdown(card1, unsafe_allow_html=True)
st.markdown(card2, unsafe_allow_html=True)
st.markdown(card3, unsafe_allow_html=True)
st.markdown(card4, unsafe_allow_html=True)
st.markdown(card5, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
