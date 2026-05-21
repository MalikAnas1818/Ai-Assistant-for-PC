import os
import subprocess
import time


# =========================
# SHUTDOWN PC
# =========================

def shutdown_pc():

    print("⚠️ Shutting down PC...")

    os.system("shutdown /s /t 5")


# =========================
# RESTART PC
# =========================

def restart_pc():

    print("🔄 Restarting PC...")

    os.system("shutdown /r /t 5")


# =========================
# LOCK PC
# =========================

def lock_pc():

    print("🔒 Locking PC...")

    os.system("rundll32.exe user32.dll,LockWorkStation")


# =========================
# OPEN CMD IN FOLDER
# =========================

def open_cmd_in_folder(folder_path):

    try:

        if not os.path.exists(folder_path):
            print("❌ Folder not found")
            return False

        print(f"💻 Opening CMD in: {folder_path}")

        os.system(f'start cmd /K cd /d "{folder_path}"')

        return True

    except Exception as e:
        print("❌ CMD error:", e)
        return False


# =========================
# CREATE VIRTUAL ENV
# =========================

def create_venv(folder_path, venv_name="venv"):

    try:

        if not os.path.exists(folder_path):
            print("❌ Folder not found")
            return False

        venv_path = os.path.join(folder_path, venv_name)

        print("🐍 Creating virtual environment...")

        # run command
        subprocess.run(
            f'python -m venv "{venv_path}"',
            shell=True
        )

        print(f"✅ Virtual env created at: {venv_path}")

        return venv_path

    except Exception as e:
        print("❌ Venv error:", e)
        return False


# =========================
# ACTIVATE VENV IN CMD
# =========================

def activate_venv(folder_path, venv_name="venv"):

    try:

        venv_activate = os.path.join(folder_path, venv_name, "Scripts", "activate")

        if not os.path.exists(venv_activate + ".bat"):
            print("❌ Venv not found")
            return False

        print("⚡ Activating virtual environment...")

        os.system(f'start cmd /K "{venv_activate}.bat"')

        return True

    except Exception as e:
        print("❌ Activate error:", e)
        return False


# =========================
# RUN COMMAND IN FOLDER CMD
# =========================

def run_command_in_folder(folder_path, command):

    try:

        if not os.path.exists(folder_path):
            print("❌ Folder not found")
            return False

        print(f"🚀 Running command: {command}")

        os.system(f'start cmd /K "cd /d {folder_path} && {command}"')

        return True

    except Exception as e:
        print("❌ Command error:", e)
        return False


# =========================
# TEST
# =========================

if __name__ == "__main__":

    test_folder = "E:/MyAIProject"

    open_cmd_in_folder(test_folder)

    create_venv(test_folder)

    time.sleep(2)

    activate_venv(test_folder)