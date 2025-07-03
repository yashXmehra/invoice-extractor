import streamlit as st
import time
from datetime import datetime

class SimpleAuth:
    def __init__(self):
        # Single user credentials - change these as needed
        self.USERNAME = "admin"
        self.PASSWORD = "invoice2024"
        
        # Session timeout in seconds (30 minutes)
        self.session_timeout = 1800
    
    def check_credentials(self, username, password):
        """Check if username and password are valid"""
        return username == self.USERNAME and password == self.PASSWORD
    
    def login(self, username, password):
        """Authenticate user and create session"""
        if self.check_credentials(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.login_time = time.time()
            return True
        return False
    
    def logout(self):
        """Logout user and clear session"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.login_time = None
    
    def is_authenticated(self):
        """Check if user is authenticated and session is valid"""
        if not hasattr(st.session_state, 'authenticated') or not st.session_state.authenticated:
            return False
        
        # Check session timeout
        if hasattr(st.session_state, 'login_time') and st.session_state.login_time:
            if time.time() - st.session_state.login_time > self.session_timeout:
                self.logout()
                return False
        
        return True
    
    def get_current_user(self):
        """Get current logged-in user"""
        if self.is_authenticated():
            return st.session_state.username
        return None
    
    def get_login_time(self):
        """Get login time"""
        if self.is_authenticated() and hasattr(st.session_state, 'login_time'):
            return st.session_state.login_time
        return None

class LoginUI:
    def __init__(self, auth):
        self.auth = auth
    
    def show_login_form(self):
        """Display the login form"""
        st.title("🔐 Invoice Extractor - Login")
        
        # Custom CSS for login form
        st.markdown("""
        <style>
        .login-form {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .login-title {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form", clear_on_submit=True):
            st.markdown("### Please sign in to access the Invoice Extractor")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                username = st.text_input(
                    "Username", 
                    placeholder="Enter your username",
                    key="login_username"
                )
                password = st.text_input(
                    "Password", 
                    type="password", 
                    placeholder="Enter your password",
                    key="login_password"
                )
                
                submitted = st.form_submit_button("🚀 Sign In", use_container_width=True)
                
                if submitted:
                    if username and password:
                        if self.auth.login(username, password):
                            st.success("✅ Login successful! Redirecting...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                    else:
                        st.warning("⚠️ Please enter both username and password")
        
        # Credentials info
        self.show_credentials_info()
    
    def show_credentials_info(self):
     """Show login help without revealing credentials"""
    with st.expander("ℹ️ Login Help"):
        st.markdown("""
        **Access Information:**
        - Contact your administrator for login credentials
        - Ensure you have proper authorization to access this system
        
        *Note: This system is for authorized users only.*
        """)
    
    def show_user_info(self):
        """Show user information in sidebar"""
        if self.auth.is_authenticated():
            current_user = self.auth.get_current_user()
            login_time = self.auth.get_login_time()
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 👤 User Session")
            st.sidebar.markdown(f"**Logged in as:** {current_user}")
            
            if login_time:
                login_str = datetime.fromtimestamp(login_time).strftime('%H:%M:%S')
                st.sidebar.markdown(f"**Login time:** {login_str}")
            
            st.sidebar.markdown("**Status:** 🟢 Active")
    
    def show_logout_button(self):
        """Show logout button in sidebar"""
        if self.auth.is_authenticated():
            st.sidebar.markdown("---")
            if st.sidebar.button("🚪 Logout", use_container_width=True):
                self.auth.logout()
                st.rerun()

# Initialize authentication system
def init_auth():
    """Initialize authentication system"""
    if 'simple_auth' not in st.session_state:
        st.session_state.simple_auth = SimpleAuth()
    
    if 'login_ui' not in st.session_state:
        st.session_state.login_ui = LoginUI(st.session_state.simple_auth)
    
    return st.session_state.simple_auth, st.session_state.login_ui

# Decorator for protecting functions
def require_auth(auth):
    """Decorator to require authentication for functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not auth.is_authenticated():
                return False
            return func(*args, **kwargs)
        return wrapper
    return decorator