import streamlit as st
from utils import init_user_settings, infer_smtp_settings


def render_settings_sidebar():
    """Render the settings in the sidebar"""
    st.sidebar.header("Settings")

    # Initialize user settings
    init_user_settings()

    # Create an expander for settings
    with st.sidebar.expander("Email Settings", expanded=False):
        # Email credentials
        previous_email = st.session_state.user_email
        email = st.text_input(
            "Your Email Address",
            value=st.session_state.user_email,
            help="Enter your email address",
            key="sidebar_email",
        )

        # If email changed, try to infer SMTP settings
        if email != previous_email and "@" in email:
            smtp_server, smtp_port = infer_smtp_settings(email)
            if smtp_server and smtp_port:
                st.session_state.smtp_server = smtp_server
                st.session_state.smtp_port = smtp_port
                st.success(
                    f"SMTP settings automatically configured for {email.split('@')[1]}"
                )

        password = st.text_input(
            "Email Password",
            value=st.session_state.user_password,
            type="password",
            help="Enter your email password",
            key="sidebar_password",
        )

        # SMTP settings
        smtp_server = st.text_input(
            "SMTP Server",
            value=st.session_state.smtp_server,
            help="SMTP server address (e.g., smtp-mail.outlook.com)",
            key="sidebar_smtp_server",
        )

        smtp_port = st.number_input(
            "SMTP Port",
            value=st.session_state.smtp_port,
            min_value=1,
            max_value=65535,
            help="SMTP server port (e.g., 587 for TLS)",
            key="sidebar_smtp_port",
        )

        # User name
        name = st.text_input(
            "Your Name",
            value=st.session_state.user_name,
            help="This will appear in the signature of your emails",
            key="sidebar_name",
        )

        # Save button
        if st.button("Save Settings", key="sidebar_save"):
            # Update session state with new values
            st.session_state.user_email = email
            st.session_state.user_password = password
            st.session_state.user_name = name
            st.session_state.smtp_server = smtp_server
            st.session_state.smtp_port = int(smtp_port)
            st.session_state.settings_saved = True

            st.success("Settings saved successfully!")

    # Test connection button in a separate expander
    with st.sidebar.expander("Test Connection", expanded=False):
        if st.button("Test Email Connection", key="sidebar_test"):
            if not st.session_state.user_email or not st.session_state.user_password:
                st.error("Please enter your email and password first")
            else:
                try:
                    import smtplib

                    # Try to connect to the SMTP server
                    with smtplib.SMTP(
                        st.session_state.smtp_server, st.session_state.smtp_port
                    ) as server:
                        server.starttls()
                        server.login(
                            st.session_state.user_email, st.session_state.user_password
                        )

                    st.success(
                        "Connection successful! Your email credentials are working."
                    )
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
                    st.markdown(
                        """
                    Common issues:
                    - Incorrect email or password
                    - Wrong SMTP server or port
                    - Two-factor authentication enabled without using an app password
                    - Account security settings blocking the connection
                    """
                    )


def render_settings_page():
    """Render the settings page for user configuration"""
    st.title("Email Sender Settings")

    # Initialize user settings
    init_user_settings()

    # Create a form for settings
    with st.form("settings_form"):
        st.subheader("Email Configuration")

        # Email credentials
        previous_email = st.session_state.user_email
        email = st.text_input(
            "Your Email Address",
            value=st.session_state.user_email,
            help="Enter your email address",
        )

        # If email changed, try to infer SMTP settings
        if email != previous_email and "@" in email:
            smtp_server, smtp_port = infer_smtp_settings(email)
            if smtp_server and smtp_port:
                st.session_state.smtp_server = smtp_server
                st.session_state.smtp_port = smtp_port
                st.success(
                    f"SMTP settings automatically configured for {email.split('@')[1]}"
                )

        password = st.text_input(
            "Email Password",
            value=st.session_state.user_password,
            type="password",
            help="Enter your email password",
        )

        # SMTP settings
        col1, col2 = st.columns(2)
        with col1:
            smtp_server = st.text_input(
                "SMTP Server",
                value=st.session_state.smtp_server,
                help="SMTP server address (e.g., smtp-mail.outlook.com)",
            )

        with col2:
            smtp_port = st.number_input(
                "SMTP Port",
                value=st.session_state.smtp_port,
                min_value=1,
                max_value=65535,
                help="SMTP server port (e.g., 587 for TLS)",
            )

        st.info(
            """
        Common SMTP Servers:
        - Outlook: smtp-mail.outlook.com:587
        - Gmail: smtp.gmail.com:587
        - Yahoo: smtp.mail.yahoo.com:587
        """
        )

        st.markdown(
            """
        > **Note**: For security, we recommend using an [App Password](https://support.microsoft.com/en-us/account-billing/manage-app-passwords-for-two-step-verification-d6dc8c6d-4bf7-4851-ad95-6d07799387e9) instead of your regular password.
        """
        )

        st.subheader("Personal Information")

        # User name
        name = st.text_input(
            "Your Name",
            value=st.session_state.user_name,
            help="This will appear in the signature of your emails",
        )

        # Submit button
        submitted = st.form_submit_button("Save Settings")

        if submitted:
            # Update session state with new values
            st.session_state.user_email = email
            st.session_state.user_password = password
            st.session_state.user_name = name
            st.session_state.smtp_server = smtp_server
            st.session_state.smtp_port = int(smtp_port)
            st.session_state.settings_saved = True

            st.success("Settings saved successfully!")

    # Test connection button
    if st.button("Test Email Connection"):
        if not st.session_state.user_email or not st.session_state.user_password:
            st.error("Please enter your email and password first")
        else:
            try:
                import smtplib

                # Try to connect to the SMTP server
                with smtplib.SMTP(
                    st.session_state.smtp_server, st.session_state.smtp_port
                ) as server:
                    server.starttls()
                    server.login(
                        st.session_state.user_email, st.session_state.user_password
                    )

                st.success("Connection successful! Your email credentials are working.")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
                st.markdown(
                    """
                Common issues:
                - Incorrect email or password
                - Wrong SMTP server or port
                - Two-factor authentication enabled without using an app password
                - Account security settings blocking the connection
                """
                )


if __name__ == "__main__":
    render_settings_page()
