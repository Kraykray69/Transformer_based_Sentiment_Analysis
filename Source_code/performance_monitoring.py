import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Monitor performance function
def monitor_performance(current_accuracy, threshold=0.85):
    if current_accuracy < threshold:
        send_email_alert("Model performance dropped! Current accuracy:", current_accuracy)
        return True
    return False

# Email alert function
def send_email_alert(message):
    # Note to self: Remember to use environment variables for these!
    sender_email = "rupeshs2103@gmail.com"
    receiver_email = "rupesh2103033@gmail.com"
    password = "bqviuuaefhfmrycc"
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Model Performance Alert"
    
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
        print("Email alert sent!")
    except Exception as e:
        print(f"Oops! Couldn't send email: {e}")