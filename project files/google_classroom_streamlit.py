import streamlit as st
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

CLIENT_SECRETS_FILE = "client_secret.json"

SCOPES = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

if "credentials" not in st.session_state:
    st.session_state["credentials"] = None

def login_with_popup():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    return creds

def main():
    st.title("✅ EduTutor - Google Classroom Sync")

    if st.session_state["credentials"] is None:
      if st.button("🔐 Login with Google"):
        creds = login_with_popup()
        st.session_state["credentials"] = creds
        st.success("✅ Logged in successfully!")
        st.rerun()  # ✅ Use this instead of experimental_rerun


    else:
        creds = st.session_state["credentials"]
        st.success("✅ Logged in with Google!")

        try:
            service = build("classroom", "v1", credentials=creds)
            results = service.courses().list(pageSize=10).execute()
            courses = results.get("courses", [])

            if not courses:
                st.warning("No courses found.")
            else:
                st.subheader("📘 Your Courses:")
                for course in courses:
                    st.write(f"➡️ {course['name']} ({course['id']})")

                    try:
                        materials = service.courses().courseWorkMaterials().list(courseId=course['id']).execute()
                        for item in materials.get("courseWorkMaterial", []):
                            st.write(f"📝 {item.get('title', 'Untitled')}")
                    except Exception as e:
                        st.info("No course materials found.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()






