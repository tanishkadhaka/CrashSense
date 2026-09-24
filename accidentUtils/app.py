import os
from object_tracker import accidentDetection
import streamlit as st
import streamlit_authenticator as stauth
import smtplib, ssl
import moviepy.editor as moviepy
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf

####################--Load Model--#############
weights = "/content/project/checkpoints/yolov4-416"
@st.cache
def load_model():
	  return tf.saved_model.load(weights, tags=[tag_constants.SERVING])
##########################################



st.title("Car collision detection")

# ---- Demo login users ----
# Configure via environment variables instead of hardcoding real names/emails/passwords.
# Example (set before running `streamlit run app.py`):
#   export APP_USER_NAMES="Demo User"
#   export APP_USER_EMAILS="demo.user@example.com"
#   export APP_USER_PASSWORDS="changeme"
names = os.environ.get("APP_USER_NAMES", "Demo User").split(",")
Emails = os.environ.get("APP_USER_EMAILS", "demo.user@example.com").split(",")
passwords = os.environ.get("APP_USER_PASSWORDS", "changeme").split(",")

hashed_passwords = stauth.hasher(passwords).generate()
authenticator = stauth.authenticate(names, Emails, hashed_passwords,
    'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)
name, authentication_status = authenticator.login('Login', 'main')
if authentication_status:
    st.write('Welcome *%s*' % (name))
    st.title('Let\'s detect any collisions')
    Url = st.text_input("Add Url Here")
    result = ""
    #Text box to recieve Url
    if st.button("Submit"):
      result = Url
      st.success(result)

      #########- Call our Function-##############

      Output = accidentDetection(result, "/content/infer1.mp4", load_model())
      Output
      ######-Print video result-##########

      #############-Convert Video to mp4-###########
      clip = moviepy.VideoFileClip("/content/infer1.mp4")
      clip.write_videofile("/content/infer2.mp4")

      #############-Display Video-#####
      video_file = open("/content/infer2.mp4", 'rb')
      video_bytes = video_file.read()
      st.video(video_bytes)

      ##############-Send Email-########
      # Set these as environment variables — never hardcode credentials here.
      #   export SENDER_EMAIL="your-email@example.com"
      #   export SENDER_EMAIL_PASSWORD="your-app-password"   # use a Gmail App Password, not your real password
      sender = os.environ.get("SENDER_EMAIL")
      password = os.environ.get("SENDER_EMAIL_PASSWORD")
      port = 465
      receiver = Emails[names.index(name)]

      if not sender or not password:
          st.warning("Email alert skipped: SENDER_EMAIL / SENDER_EMAIL_PASSWORD environment variables are not set.")
      else:
          server = smtplib.SMTP('smtp.gmail.com', 587)
          server.starttls()
          server.login(sender, password)
          server.sendmail(sender, receiver, Output)
          server.quit()



elif authentication_status == False:
    st.error('Email/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your Email and password')
