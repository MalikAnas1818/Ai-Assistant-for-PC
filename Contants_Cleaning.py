import csv
import re


def clean_name(name):

    # remove emojis + special chars
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)

    return name.strip().lower()


def clean_phone(phone):

    phone = phone.strip()

    # remove spaces, dashes, brackets
    phone = re.sub(r'[^\d+]', '', phone)

    # fix Pakistan numbers
    if phone.startswith("03"):
        phone = "+92" + phone[1:]

    if not phone.startswith("+"):
        phone = "+" + phone

    return phone


def clean_contacts(input_file, output_file="data/contacts.csv"):

    cleaned = []

    with open(input_file, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            # =========================
            # SMART NAME PICKING
            # =========================

            name = ""

            # try full name fields first
            if row.get("Name"):
                name = row.get("Name")

            elif row.get("File As"):
                name = row.get("File As")

            else:
                first = row.get("First Name", "")
                middle = row.get("Middle Name", "")
                last = row.get("Last Name", "")

                name = " ".join([first, middle, last]).strip()

            # =========================
            # PHONE PICKING
            # =========================

            phone = row.get("Phone 1 - Value", "")

            # skip empty rows
            if not name or not phone:
                continue

            name = clean_name(name)
            phone = clean_phone(phone)

            cleaned.append({
                "Name": name,
                "Phone": phone
            })

    # =========================
    # WRITE CLEAN FILE
    # =========================

    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=["Name", "Phone"])
        writer.writeheader()
        writer.writerows(cleaned)

    print(f"✅ Clean CSV ready: {output_file}")
    print(f"📊 Total contacts: {len(cleaned)}")


# TEST
if __name__ == "__main__":
    clean_contacts("data/contacts.csv")