import csv
import pywhatkit as kit
import pyautogui
import time
import os


# =========================
# LOAD CONTACTS FROM CSV
# =========================

def load_contacts(file_path="data/contacts.csv"):

    contacts = {}

    try:

        if not os.path.exists(file_path):
            print("❌ contacts.csv not found")
            return {}

        with open(file_path, "r", encoding="utf-8") as file:

            reader = csv.DictReader(file)

            for row in reader:

                name = row.get("Name", "").strip().lower()
                phone = row.get("Phone", "").strip().replace(" ", "")

                if name and phone:
                    contacts[name] = phone

        print(f"✅ Contacts loaded: {len(contacts)}")

        return contacts

    except Exception as e:
        print("❌ CSV Load Error:", e)
        return {}


# =========================
# GET CONTACT NUMBER
# =========================

def get_contact_number(name):

    contacts = load_contacts()

    name = name.lower().strip()

    # direct match
    if name in contacts:
        return contacts[name]

    # fuzzy match
    for key in contacts:
        if name in key or key in name:
            return contacts[key]

    return None


# =========================
# SEND MESSAGE (STABLE FIX)
# =========================

def send_instant_message(contact_name, message):

    try:

        number = get_contact_number(contact_name)

        if not number:
            print(f"❌ Contact not found: {contact_name}")
            return False

        print(f"📲 Sending message to {contact_name} -> {number}")

        kit.sendwhatmsg_instantly(
            phone_no=number,
            message=message,
            wait_time=20,
            tab_close=False,
            close_time=5
        )

        # wait for WhatsApp fully load
        time.sleep(12)

        # FORCE FOCUS (VERY IMPORTANT)
        pyautogui.click()

        time.sleep(1)

        # 🔥 REAL FIX: USE HOTKEY INSTEAD OF ENTER
        pyautogui.hotkey("shift", "enter")  # ensures focus stable
        time.sleep(0.5)

        pyautogui.press("enter")

        print("✅ Message sent automatically")

        return True

    except Exception as e:
        print("❌ WhatsApp Error:", e)
        return False

# =========================
# DEBUG CONTACTS
# =========================

def show_contacts():

    contacts = load_contacts()

    print("\n📒 CONTACT LIST:")

    for name, number in contacts.items():
        print(f"{name} → {number}")


# =========================
# TEST
# =========================

if __name__ == "__main__":

    show_contacts()

    send_instant_message(
        "malik anas talha",
        "Hello! 🚀 AI Assistant se message aya hai"
    )
