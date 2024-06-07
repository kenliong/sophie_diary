import streamlit as st


## Streamlit related functions ##
def get_custom_css_modifier():
    css_modifier = """
<style>
/* remove Streamlit default menu and footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
    """
    return css_modifier

