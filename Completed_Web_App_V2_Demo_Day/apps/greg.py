import streamlit as st
import streamlit.components.v1 as components

def app():

    st.sidebar.write("Does an event like the closing of Turner Field and the opening of SunTrust Park affect housing near those locations?  Available data didn't provide strong support of that.")

    greg_temp = """<div class='tableauPlaceholder' id='viz1614827172701' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;Project3_16143650057100&#47;TurnerandSuntrust&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Project3_16143650057100&#47;TurnerandSuntrust' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;Project3_16143650057100&#47;TurnerandSuntrust&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1614827172701');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1280px';vizElement.style.height='814px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(greg_temp, height = 1200, width = 2000)


    